mod model;

use make87::encodings::{Encoder, ProtobufEncoder};
use make87::interfaces::zenoh::{ConfiguredProvider, ConfiguredSubscriber, ZenohInterface};
use make87::models::ApplicationConfig;
use make87_messages::image::uncompressed::ImageRawAny;
use make87_messages::detection::r#box::Boxes2DAxisAligned;
use model::Model;
use std::error::Error;
use std::sync::Arc;
use make87::config::load_config_from_default_env;
use tokio::sync::Mutex;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    loop {
        let res = run_loop().await;
        if let Err(e) = res {
            eprintln!("Error in run_loop: {}", e);
            // sleep for a while before retrying
            tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
        }
    }

    Ok(())

}



async fn run_loop() -> Result<(), Box<dyn Error + Send + Sync>> {
    let model_path = "/models/model.onnx";
    let config = load_config_from_default_env()?;
    let app_config = config.config;
    let num_threads = app_config.get("num_threads")
        .and_then(|v| v.as_u64())
        .unwrap_or(1) as usize;
    let classes = app_config.get("classes")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect::<Vec<String>>());

    let model = Model::new(model_path, num_threads, 0.2, classes);
    let model = Arc::new(Mutex::new(model?));

    let zenoh_interface = ZenohInterface::from_default_env("zenoh-client")?;
    let session_handle = zenoh_interface.get_session().await?;

    // Subscribe to images for publish output
    if let ConfiguredSubscriber::Fifo(sub) = zenoh_interface.get_subscriber(&session_handle, "IMAGE_TOPIC").await? {
        let decoder = ProtobufEncoder::<ImageRawAny>::new();
        let encoder = ProtobufEncoder::<Boxes2DAxisAligned>::new();
        let publisher = zenoh_interface.get_publisher(&session_handle, "DETECTIONS_TOPIC").await?;

        let model_clone = model.clone();
        tokio::spawn(async move {
            while let Ok(sample) = sub.recv_async().await {
                if let Ok(image) = decoder.decode(&sample.payload().to_bytes()) {
                    let mut model_guard = model_clone.lock().await;
                    match model_guard.predict(&image).await {
                        Ok(detections) => {
                            let encoded = encoder.encode(&detections).unwrap();
                            if let Err(e) = publisher.put(&encoded).await {
                                eprintln!("Failed to publish: {}", e);
                            }
                        }
                        Err(e) => {
                            eprintln!("Inference error: {}", e);
                        }
                    }
                }
            }
        });
    }

    // Provider setup
    if let ConfiguredProvider::Fifo(provider) = zenoh_interface.get_provider(&session_handle, "IMAGE_TOPIC").await? {
        let decoder = ProtobufEncoder::<ImageRawAny>::new();
        let encoder = ProtobufEncoder::<Boxes2DAxisAligned>::new();
        let model_clone = model.clone();

        while let Ok(query) = provider.recv_async().await {
            if let Some(payload) = query.payload() {
                if let Ok(image) = decoder.decode(&payload.to_bytes()) {
                    let mut model_guard = model_clone.lock().await;
                    let detections = model_guard.predict(&image).await.unwrap_or_default();
                    let encoded = encoder.encode(&detections)?;
                    query.reply(&query.key_expr().clone(), &encoded).await?;
                }
            }
        }
    }
    make87::run_forever();

    Ok(())
}