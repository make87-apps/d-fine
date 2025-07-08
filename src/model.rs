use image::{Rgb, RgbImage};
use make87_messages::detection::r#box::Boxes2DAxisAligned;
use make87_messages::image::uncompressed::ImageRawAny;
use ndarray::{Array, Array3, Array4, ArrayViewMut3};
use ort::session::Session;
use ort::value::{Tensor, Value};
use std::collections::{HashMap, HashSet};
use std::error::Error;
// import only when gpu feature is enabled
#[cfg(feature = "gpu")]
use ort::execution_providers::CUDAExecutionProvider;

pub struct Model {
    pub confidence_threshold: f32,
    pub active_label_ids: HashSet<usize>,
    pub session: Session,
}

impl Model {
    pub fn new(
        weights_path: &str,
        num_threads: usize,
        confidence_threshold: f32,
        classes: Option<Vec<String>>,
    ) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let mut builder = Session::builder()?;

        #[cfg(feature = "gpu")]
        {
            builder =
                builder.with_execution_providers([CUDAExecutionProvider::default().build()])?;
        }

        // Only apply intra-thread setting if GPU is not used
        #[cfg(not(feature = "gpu"))]
        {
            let _ = num_threads; // silence unused warning
            builder = builder.with_intra_threads(num_threads)?;
        }

        let session = builder.commit_from_file(weights_path)?;

        let full_label_list = coco91_labels();
        // if classes is NOne use all classes
        let active_label_ids = match &classes {
            None => full_label_list
                .iter()
                .enumerate()
                .map(|(i, name)| i)
                .collect::<HashSet<usize>>(),
            Some(c) => {
                let label_index_map: HashMap<String, usize> = full_label_list
                    .iter()
                    .enumerate()
                    .map(|(i, name)| (name.clone(), i))
                    .collect();
                let active_label_ids: HashSet<usize> = c
                    .iter()
                    .filter_map(|cls| label_index_map.get(cls).copied())
                    .collect();
                active_label_ids
            }
        };

        //
        //
        //
        //

        Ok(Model {
            confidence_threshold,
            active_label_ids,
            session,
        })
    }

    pub async fn predict(
        &mut self,
        image: &ImageRawAny,
    ) -> Result<Boxes2DAxisAligned, Box<dyn Error + Send + Sync>> {
        let rgb_img = image_raw_any_to_rgb_image(image).ok_or("Failed to convert image")?;
        let (resized, ratio, pad_w, pad_h) = resize_with_aspect_ratio(&rgb_img, 640);
        let input_tensor = rgbimage_to_tensor(&resized);

        let orig_h = rgb_img.height() as f32;
        let orig_w = rgb_img.width() as f32;
        let orig_target_sizes = ndarray::arr2(&[[orig_h as i64, orig_w as i64]]);

        let input_tensor = Tensor::from_array(input_tensor)?;
        let orig_target_sizes = Tensor::from_array(orig_target_sizes)?;

        let outputs = self.session.run(ort::inputs![
            "images" => input_tensor,
            "orig_target_sizes" => orig_target_sizes
        ])?;

        let (box_shape, box_data) = outputs["boxes"].try_extract_tensor::<f32>()?;
        let (_, label_data) = outputs["labels"].try_extract_tensor::<i64>()?;
        let (_, score_data) = outputs["scores"].try_extract_tensor::<f32>()?;

        let mut detections = Vec::new();
        let num_boxes = box_shape[0] as usize;
        // let stride = box_shape[1]; // should be 4

        for i in 0..num_boxes {
            let score = score_data[i];
            let label_id = label_data[i] as usize;
            if score < self.confidence_threshold || !self.active_label_ids.contains(&label_id) {
                continue;
            }

            let x0 = (box_data[i * 4 + 0] - pad_w as f32) / ratio;
            let y0 = (box_data[i * 4 + 1] - pad_h as f32) / ratio;
            let x1 = (box_data[i * 4 + 2] - pad_w as f32) / ratio;
            let y1 = (box_data[i * 4 + 3] - pad_h as f32) / ratio;

            detections.push(make87_messages::detection::r#box::Box2DAxisAligned {
                header: None,
                geometry: Some(make87_messages::geometry::r#box::Box2DAxisAligned {
                    header: None,
                    x: x0,
                    y: y0,
                    width: x1 - x0,
                    height: y1 - y0,
                }),
                confidence: score,
                class_id: label_id as i32,
            });
        }

        Ok(make87_messages::detection::r#box::Boxes2DAxisAligned {
            header: None,
            boxes: detections,
        })
    }
}

/// Resize with aspect ratio and pad to square (like PIL version)
fn resize_with_aspect_ratio(image: &RgbImage, target_size: u32) -> (RgbImage, f32, u32, u32) {
    let (orig_w, orig_h) = image.dimensions();
    let scale = (target_size as f32 / orig_w as f32).min(target_size as f32 / orig_h as f32);
    let new_w = (orig_w as f32 * scale).round() as u32;
    let new_h = (orig_h as f32 * scale).round() as u32;

    let resized =
        image::imageops::resize(image, new_w, new_h, image::imageops::FilterType::Triangle);

    let mut padded = RgbImage::new(target_size, target_size);
    let pad_w = (target_size - new_w) / 2;
    let pad_h = (target_size - new_h) / 2;

    image::imageops::overlay(&mut padded, &resized, pad_w.into(), pad_h.into());

    (padded, scale, pad_w, pad_h)
}

/// Convert RgbImage to normalized CHW format (1, 3, H, W)
fn rgbimage_to_tensor(image: &RgbImage) -> Array4<f32> {
    let (w, h) = image.dimensions();
    let mut array = Array::zeros((1, 3, h as usize, w as usize));
    for (x, y, pixel) in image.enumerate_pixels() {
        let [r, g, b] = pixel.0;
        array[[0, 0, y as usize, x as usize]] = r as f32 / 255.0;
        array[[0, 1, y as usize, x as usize]] = g as f32 / 255.0;
        array[[0, 2, y as usize, x as usize]] = b as f32 / 255.0;
    }
    array
}

pub fn image_raw_any_to_rgb_image(image: &ImageRawAny) -> Option<RgbImage> {
    match &image.image {
        Some(make87_messages::image::uncompressed::image_raw_any::Image::Rgb888(rgb)) => {
            let (w, h) = (rgb.width as u32, rgb.height as u32);
            if rgb.data.len() != (w * h * 3) as usize {
                println!("RGB888 size mismatch");
                return None;
            }

            let mut img = RgbImage::new(w, h);
            for y in 0..h {
                for x in 0..w {
                    let i = ((y * w + x) * 3) as usize;
                    img.put_pixel(x, y, Rgb([rgb.data[i], rgb.data[i + 1], rgb.data[i + 2]]));
                }
            }
            Some(img)
        }

        Some(make87_messages::image::uncompressed::image_raw_any::Image::Rgba8888(rgba)) => {
            let (w, h) = (rgba.width as u32, rgba.height as u32);
            if rgba.data.len() != (w * h * 4) as usize {
                println!("RGBA8888 size mismatch");
                return None;
            }

            let mut img = RgbImage::new(w, h);
            for y in 0..h {
                for x in 0..w {
                    let i = ((y * w + x) * 4) as usize;
                    img.put_pixel(
                        x,
                        y,
                        Rgb([rgba.data[i], rgba.data[i + 1], rgba.data[i + 2]]),
                    );
                }
            }
            Some(img)
        }

        Some(make87_messages::image::uncompressed::image_raw_any::Image::Nv12(nv12)) => {
            let (w, h) = (nv12.width as u32, nv12.height as u32);
            let y_len = (w * h) as usize;
            if nv12.data.len() < y_len {
                println!("NV12 size mismatch");
                return None;
            }

            let mut img = RgbImage::new(w, h);
            for y in 0..h {
                for x in 0..w {
                    let gray = nv12.data[(y * w + x) as usize];
                    img.put_pixel(x, y, Rgb([gray, gray, gray]));
                }
            }
            Some(img)
        }

        Some(make87_messages::image::uncompressed::image_raw_any::Image::Yuv420(yuv)) => {
            y_plane_to_rgb(&yuv.data, yuv.width, yuv.height)
        }
        Some(make87_messages::image::uncompressed::image_raw_any::Image::Yuv422(yuv)) => {
            y_plane_to_rgb(&yuv.data, yuv.width, yuv.height)
        }
        Some(make87_messages::image::uncompressed::image_raw_any::Image::Yuv444(yuv)) => {
            y_plane_to_rgb(&yuv.data, yuv.width, yuv.height)
        }
        _ => {
            println!("Unsupported format");
            None
        }
    }
}

fn y_plane_to_rgb(y_data: &[u8], width: u32, height: u32) -> Option<RgbImage> {
    let (w, h) = (width as u32, height as u32);
    if y_data.len() < (w * h) as usize {
        println!("Y plane size too small");
        return None;
    }

    let mut img = RgbImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let val = y_data[(y * w + x) as usize];
            img.put_pixel(x, y, Rgb([val, val, val]));
        }
    }
    Some(img)
}

pub fn coco91_labels() -> Vec<String> {
    vec![
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]
    .into_iter()
    .map(|s| s.to_string())
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array4;
    use rand::Rng;

    #[tokio::test]
    async fn test_model_with_random_input() -> Result<(), Box<dyn Error + Send + Sync>> {
        let weights_path = "/home/phillip/projects/D-FINE/models/dfine_s_obj2coco.onnx"; // ← replace with actual path
        let mut model = Model::new(
            weights_path,
            2,                    // num_threads
            0.5,                  // confidence_threshold
            Some(vec!["dummy".into()]), // dummy classes
        )?;

        // Generate noise image
        let mut rng = rand::thread_rng();
        let mut array = Array4::<f32>::zeros((1, 3, 640, 640));
        array.map_inplace(|x| *x = rng.gen());

        let orig_target_sizes = ndarray::arr2(&[[640_i64, 640_i64]]);
        let input_tensor = Tensor::from_array(array)?;
        let orig_tensor = Tensor::from_array(orig_target_sizes)?;

        let outputs = model.session.run(ort::inputs![
            "images" => input_tensor,
            "orig_target_sizes" => orig_tensor
        ])?;

        assert!(outputs.contains_key("boxes"));
        assert!(outputs.contains_key("scores"));
        assert!(outputs.contains_key("labels"));
        // print labels and scores as pairs not the boxes

        if let Some(labels) = outputs.get("labels") {
            // scores
            if let Some(scores) = outputs.get("scores") {
                let (_, labels_tensor) = labels.try_extract_tensor::<i64>()?;
                let (_, scores_tensor) = scores.try_extract_tensor::<f32>()?;

                for (label, score) in labels_tensor.iter().zip(scores_tensor.iter()) {
                    println!("Label: {}, Score: {}", label, score);
                }
            } else {
                println!("No scores found in outputs");
            }
        }

        Ok(())
    }
}
