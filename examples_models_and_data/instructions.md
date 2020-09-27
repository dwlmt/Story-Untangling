# Brief Instructions for running the models on other datasets

## Installation

There is a requirements.txt and environment.yml file from conda in the
main folder. Most of the environment way via the requirements file or the conda env.
A couple of libraries such as nostril need to be installed via git.
The allennlp version used is now old and the code is only tested to work on
0.8.x . 

## Model and Predictors

The model contained in this director is the main LSTM model used for 
the ACL paper. The config file for training this model is under /experiments/lstm_fusion_big_full_experiment.json .

To run inference on text the a path to the dataset needs to be specified. The main script used to run prediction is
in /slurm/predict_batch.sh to run on a slurm env. The following script is the main command.


```shell script

allennlp predict --include-package story_untangling \
    --use-dataset-reader \
    --predictor uncertain_reader_gen_predictor \
     ${CLUSTER_HOME}/${MODEL_PATH} \
     ${CLUSTER_HOME}/${DATASET_SOURCE} --cuda-device 0 \
    --output-file  ${SERIAL_DIR}/${EXP_NAME}_prediction_output.jsonl \

```

The model path parameter is the path to the zipped model file. The dataset source the path
to the text file to run on and output-file the destination for the suspense prediction. The
output format is json lines with one line per input story.

The input dataset is designed  WritingPrompts -
[download writing prompts](https://dl.fbaipublicfiles.com/fairseq/data/writingPrompts.tar.gz) .
The WritingPrompts is just a single file with one story per line and a special <newline>
token to represent real newlines in the text.

### Optional: Precache local database.

This is optional but converting to SQLLite might be better than
attempting to just run on raw text files.

The dataset is preprocessed into SQL lite database format.
The example file provided is in this directory and called 'test.wp_target_def.db.tar.gz'.

The main tables are story with id information for uniquely identifying each story and metadata like sentence length in
tokens and start and end span. The sentence table contains the text of the main table
with sentence_num indicating the ordering of the table.
The existing database contains some gaps in the sentence numbers as some  for WritingPrompts 
as some preprocessing was done to filter out long english sentences or invalid
sentences such as long sequences of punctuation.

Any preprocessed db file will be read ahead of processing the raw if the config in the model file
is updated. To do that the db needs to be in the follow location.

```python
database_file = f"{dataset_path}/{file_name}_{db_discriminator}.db"
```

The dataset path is in the config file of the model as follows.

```json
 "dataset_reader": {
    "type": "writing_prompts_whole_story",
    ...
    "dataset_path": "/afs/inf.ed.ac.uk/group/project/comics/stories/WritingPrompts/dataset_db/text",
    "use_existing_cached_db": true,
    ...
```

## Overriding options

Additional options for changing the prediction setup are provided via UNIX env variables.
These are the main ones that might be changed. Rollout levels are how steps to go ahead with
suspese. Generate and sample per branch are now many sentences to generate per step in the
beam search. The beam size if the overall beam to keep when generating forward multiple steps
into the future. PREDICTION_SAMPLE_TOP_K_WORDS is only for generated sentences and is
how many top K word tokens GPT should use when generating future sentences. Other options can be found
in the "uncertain_reader_gen_predictor"

```python
  self.levels_to_rollout = int(os.getenv('PREDICTION_LEVELS_TO_ROLLOUT', 1))
  self.generate_per_branch = int(os.getenv('PREDICTION_GENERATE_PER_BRANCH', 100)
  self.sample_per_level_branch = int(os.getenv('PREDICTION_SAMPLE_PER_BRANCH',100))
 
  self.global_beam_size =  int(os.getenv('PREDICTION_BEAM_SIZE', 10))
  self.sample_top_k_words =  int(os.getenv('PREDICTION_SAMPLE_TOP_K_WORDS', 50))
        
  self.sentiment_weighting =  float(os.getenv('PREDICTION_SENTIMENT_WEIGHTING', 1.0))
       
```

## JSON Output

The file name is the name for the txt file which for test will be "test.wp_target" 
and db_discriminator is by default def. Hence the name of the default db provided.

The output json as a list of predictions under "children" on each line.
This is an example of one of these json output lines with various predictions.

```json
{"corpus_surprise_entropy": 1.3794763088226318, "corpus_surprise_entropy_1": 1.3794763088226318, "corpus_surprise_gpt_embedding": 1.1920928955078125e-07, "corpus_surprise_l1": 53.48820877075195, "corpus_surprise_l1_1": 53.48820877075195, "corpus_surprise_l1_state": 53.48820877075195, ... "name": "3", "number_of_sentences": 66, "sentence_id": 66494, "sentence_num": 3, "sentence_text": "Might make it easier , you know ?", "sentiment": 1.4215, "steps": 1, "story_id": 1296, "textblob_sentiment": 0.0, "vader_sentiment": 0.4215}
```

## Processing results

Probably the processing of results is custom depending on use. But there are several scripts
to help with post-processing the results. 

The script predictor_extract_json_stats.py accepts the prediction json (--json-source and --output-dir for the output location) and will
out 3 compressed csvs: batch_stats which is summarised metrics for the whole story,
position_stats which is the main results, and window_stats. Window stats create sliding 
windows of various lengths rather than sentence to sentence changes but this wasn't
use in the ACL paper.

The main csv is position_stats and the main columns are story_id, sentence_num, and the individual metric columns.

The main evaluation script us story_stats_and_annotation_comp.py. This takes the --position-stats and 
--batch-stats as input. The main function of this is to generate plots for each of the stories and the suspense 
and surprise measures which also includes find peaks which will find the main peaks and troughs.
The script also has functionality via --mturk-sentence-annotations and --firebase-sentence-annotations
to fit human annotations to the suspense predictions and do inter-annotator agreement on the 
but this depends on using the same AWS collection format. It's probably simpler
just to change the script columns to match if this is required.








