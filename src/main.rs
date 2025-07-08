use make87_messages::text::PlainText;

fn main() {
    make87::initialize();

    let topic_name = "INCOMING_MESSAGE";
    match make87::resolve_topic_name(topic_name) {
        Some(topic_name) => {
            if let Some(topic) = make87::get_subscriber::<PlainText>(topic_name) {
                topic.subscribe(
                    |message| {
                        println!("Received message '{:?}'", message);
                    }
                ).unwrap();
            }
        }
        None => {
            panic!("{}", format!("Failed to resolve topic name '{}'", topic_name));
        }
    }

    make87::keep_running();
}
