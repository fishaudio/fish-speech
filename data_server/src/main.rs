use clap::Parser;
use log::info;
use prost::Message;
use rand::seq::SliceRandom;
use rand::{thread_rng, Rng};
use std::fs::File;
use std::io::{self, BufReader, Read, Result as IoResult};
use std::vec;
use tonic::{transport::Server, Request, Response, Status};

pub mod text_data {
    tonic::include_proto!("text_data");
}

use text_data::{
    data_service_server::{DataService, DataServiceServer},
    SampleDataRequest, SampledData, Sentence, TextData,
};

#[derive(Default)]
pub struct MyDataService {
    groups: Vec<TextData>,
    causual_sampling: bool,
    weights: Vec<f32>,
}

fn read_pb_stream<R: Read>(mut reader: BufReader<R>) -> io::Result<Vec<TextData>> {
    let mut text_data_list = Vec::new();
    let mut index = 0;

    loop {
        let mut size_buf = [0u8; 4];
        match reader.read_exact(&mut size_buf) {
            Ok(()) => (),
            Err(ref e) if e.kind() == io::ErrorKind::UnexpectedEof => break, // End of file
            Err(e) => return Err(e),
        }

        let size = u32::from_le_bytes(size_buf) as usize;

        let mut message_buf = vec![0u8; size];
        reader.read_exact(&mut message_buf)?;

        let text_data = TextData::decode(&message_buf[..])
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        text_data_list.push(text_data);

        index += 1;

        if index % 10000 == 0 {
            info!("Loaded {} groups", index);
        }
    }

    Ok(text_data_list)
}

impl MyDataService {
    pub fn new(files: Vec<String>, causual_sampling: bool) -> IoResult<Self> {
        let mut groups = Vec::new();
        let mut weights = Vec::new();

        for filename in files.iter() {
            let file = File::open(filename)?;
            let reader = BufReader::new(file);

            // Assuming read_pb_stream is implemented and it returns an iterator over TextData
            for text_data in read_pb_stream(reader)? {
                groups.push(text_data.clone());
                weights.push(text_data.sentences.len() as f32); // Assuming sentences is a repeated field in TextData
            }
        }

        info!("Loaded {} groups", groups.len());

        Ok(MyDataService { groups, weights, causual_sampling })
    }
}

#[tonic::async_trait]
impl DataService for MyDataService {
    async fn sample_data(
        &self,
        request: Request<SampleDataRequest>,
    ) -> Result<Response<SampledData>, Status> {
        let mut num_samples = request.into_inner().num_samples as usize;
        let mut rng = thread_rng();

        let group = self
            .groups
            .choose_weighted(&mut rng, |item| item.sentences.len() as f32);

        if group.is_err() {
            return Err(Status::internal("Failed to select a group"));
        }

        let group = group.unwrap();

        if self.causual_sampling {
            if num_samples > group.sentences.len() {
                num_samples = group.sentences.len();
            }

            // Random number between 0 and group.sentences.len() - num_samples
            let max = group.sentences.len() - num_samples;
            if max <= 0 {
                return Ok(Response::new(SampledData {
                    name: group.name.clone(), 
                    source: group.source.clone(),
                    samples: group.sentences.clone(),
                }));
            }

            let start = rng.gen_range(0..max);
            Ok(Response::new(SampledData {
                name: group.name.clone(), 
                source: group.source.clone(),
                samples: group.sentences[start..start + num_samples].to_vec(),
            }))
        } else {
            let sentences_ref = group
                .sentences
                .choose_multiple(&mut rng, num_samples);

            let sentences: Vec<Sentence> = sentences_ref
                .into_iter()
                .cloned() // Clone each &Sentence to get Sentence
                .collect();

            Ok(Response::new(SampledData {
                name: group.name.clone(), 
                source: group.source.clone(),
                samples: sentences,
            }))
        }
    }
}

/// My Data Service Application
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Files to process
    #[clap(short, long, value_name = "FILE", required = true)]
    files: Vec<String>,

    /// Causual sampling
    #[clap(short, long, default_value = "false")]
    causal: bool
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    // Parse command-line arguments
    let args = Args::parse();
    info!("Arguments: {:?}", args);

    let addr = "127.0.0.1:50051".parse()?;
    let data_service = MyDataService::new(args.files, args.causal)?;

    info!("Starting server at {}", addr);

    Server::builder()
        .add_service(DataServiceServer::new(data_service))
        .serve(addr)
        .await?;

    Ok(())
}
