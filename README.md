# VeriTaS: The First Dynamic Benchmark for Multimodal Automated Fact-Checking
![Teaser Figure](teaser.jpg)

This is the official code repository of the VeriTaS benchmark, see the [project page](https://veritas.mai.informatik.tu-darmstadt.de/) for the paper, the data, leaderboards, and more. 

The code is actively used to gather the data for the VeriTaS benchmark, including future quarters. The repository will be maintained until at least the end of 2028.

> [!NOTE]  
> If you want to use the VeriTaS benchmark data, you can apply for access [here](https://veritas.mai.informatik.tu-darmstadt.de/). Executing the pipeline on your own is not needed (but you are free to do so).


## Setup
> [!NOTE]  
> We recommend using Conda.

1. Clone the repository:
   ```bash
   git clone https://github.com/multimodal-ai-lab/Veritas.git
   ```
2. Install the dependencies:
   ```bash
   cd VeriTaS
   conda env create -n veritas python=3.12
   conda activate veritas
   pip install -e .
   ```
3. Install FFmpeg for video processing:
   ```bash
   conda install -c conda-forge ffmpeg
   ```
4. Install Playwright for powerful scraping:
   ```bash
    conda config --add channels conda-forge
    conda config --add channels microsoft
    conda install playwright
    playwright install
   ```
5. Configure the environment variables in `config.yaml`. Required variables:
    - `openai`: The OpenAI API key.
    - `google`: The Google API key.
    - `anthropic`: The Anthropic API key.
    - `database`: The credentials of the DB where to store all pipeline data.
    - `selfhosted.url`: The URL for self-hosted LLMs.
6. Configure the secrets needed by `scrapeMM` for scraping social media services:
   ```bash
   python -m scrapemm
   ```


## Collecting a New Quarter
VeriTaS is organized in quarterly splits, extended dynamically in the future with new quarters. After a calendar quarter has ended, follow these steps to generate a new quarter split:
1. **Prepare services**: Make sure all required APIs and other external services are accessible and running. See below for the list of required services. Also, make sure you have the needed funding.
2. Run the test files of `scrapeMM` to make sure everything works as expected.
3. **Update signatories (publishers)**: Run
   ```bash
   python -m scripts.update_signatories
   ```
4. **Run the pipeline**: Configure the pipeline parameters in `config.yaml` as needed and run
   ```bash
   python -m scripts.run_pipeline
   ```
5. **Export the data**: Create an export archive file of the quarter by adjusting the script `scripts/export/export.py` and running it.
6. **Create backup**: Run `python -m scripts.export.create_backup` to create a backup of the entire pipeline data.


## Cleaning Up the Media Store
The ezMM store keeps every medium the pipeline ever downloaded, including those of dismissed reviews, re-scraped appearances and articles that never made it into a claim. To reclaim that space:

```bash
python -m scripts.cleanup_media           # report only, touches no file
python -m scripts.cleanup_media --apply   # move the unreferenced files
```

The script first rebuilds the `media` table from every `<image:42>`-style reference in the claims, appearances, articles and gold evidence, recording for each medium the IDs it occurs under. It then compares that index against the store and **moves** — never deletes — every unreferenced file into `<ezmm_path>/to-delete`, mirroring its path below the store root, so a cleanup can be undone by moving the files back. Both steps report statistics (media per kind, files and bytes kept and freed, broken references); `--report <path>` additionally writes them as JSON.

Note that the ezMM registry (`item_registry.db`) is deliberately left untouched, so its rows for quarantined media become dangling until the files are deleted for good.

The report also states the size of ezMM's staging directory `<ezmm_path>/items`. ezMM stages every download there and then *copies* — rather than moves — it into `image/`, `video/` or `audio/`, so that directory holds a duplicate of nearly every medium. The script does not touch it: a file there may still be in flight, and an item that was never relocated has its registered path pointing at it. Freeing that space needs a separate, registry-aware pass.


## Gold Evidence Reconstruction
A separate analysis pipeline reconstructs the evidence the original professional fact-check used, filters invalid and leaked evidence, and validates whether the remainder suffices to recover the VeriTaS gold verdict. It answers whether evidence that only became available *during* the fact-checking period is necessary for that reconstruction.

```bash
python -m scripts.gold_evidence.run_reconstruction   # parameters set in config.yaml
python -m scripts.gold_evidence.run_temporal_analysis
```

It runs independently of the 7-stage benchmark pipeline and never modifies existing data — see [`veritas/gold_evidence/README.md`](veritas/gold_evidence/README.md) for usage and [`veritas/gold_evidence/DESIGN_DECISIONS.md`](veritas/gold_evidence/DESIGN_DECISIONS.md) for the methodological choices.

### Web UI
A read-only web interface for browsing the reconstructed evidence — including its images and videos — and the aggregate statistics:

```bash
docker compose up webui   # -> http://localhost:8080
```

See [`webui/README.md`](webui/README.md) for configuration.


## Required Services
- LLM APIs:
  - OpenAI
  - Anthropic
  - Google Gemini
- Scraping:
  - X (Twitter)
  - TikTok
  - Decodo
  - Firecrawl (can set up self-hosted instance)

_Optional_:
- Facebook (non-API scraping does a good job already)
- Telegram (not many appearances here)
- Bluesky (not many appearances here)


## TODO
- Improve performance/speed of the pipeline
- Save all model outputs and reasonings to the DB


## Acknowledgements
We thank [Martino Mensio](https://github.com/MartinoMensio) for providing useful [ClaimReview collection scripts](https://github.com/MartinoMensio/claimreview-collector/tree/b4c7bbf81673de1d4b0cd96528ca96921bd002e3) that were partially reused within `veritas/pipeline/util`. The respective code snippets are licensed under the [CC BY-NC-SA 4.0 License](https://creativecommons.org/licenses/by-nc-sa/4.0/).


## License
This repository is licensed under the [Apache 2.0 License](LICENSE).
