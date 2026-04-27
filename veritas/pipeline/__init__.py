from .main import run_pipeline
from veritas import globals

pipeline_globals = globals.get("pipeline").get("global")
min_scraped_article_length = pipeline_globals.get("min_scraped_article_length")
min_extracted_article_length = pipeline_globals.get("min_extracted_article_length")
max_extracted_article_length = pipeline_globals.get("max_extracted_article_length")
max_appearances_per_claim = pipeline_globals.get("max_appearances_per_claim")
max_video_size = pipeline_globals.get("max_video_size")
max_video_duration = pipeline_globals.get("max_video_duration")
max_media_per_claim = pipeline_globals.get("max_media_per_claim")
n_frames_per_video = pipeline_globals.get("n_frames_per_video")
sufficient_agreements_threshold = pipeline_globals.get("sufficient_agreements_threshold", 1)
