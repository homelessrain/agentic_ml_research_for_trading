# Agentic ML Research for Trading

## The What

The goal of this repo is to build an agent-assisted framework to facilitate ML research for trading purposes.

This is NOT a backtesting framework, or a technical indicator / trading signal library, or a generic ML library.


## The Why

1. **Applying ML in financial markets is extremely hard**. We need a lot of ML iterations and constantly recap learnings so far and adjust future directions accordingly. And this process needs to repeat as markets shift constantly. We need to think very holistically, including label definition, ML feasibility, feature engineering, and training data selection. If we can solve ML for trading with an agentic framework, then we should be able to use this framework for many other use cases as well.


2. **ML research process is manual**. In practice, we spend more time in building the ML infra than thinking deep about ML iterations. The more we can automate the ML research process, the more we can actually spend time in solving the modeling problem.


## The How

### Repo Structure
* `models`: ML model implementations. It will have a sub-folder called `agent`, which contains the model files added by agent
* `scripts`: any reusable scripts for training, evaluation, and data visualization
* `utils`: any sharable code components across entire repo
* `reports`: the folder that contains your research
* `recaps`: research recaps/reflections will be kept in this folder. Recaps are very useful to reason about past research results and determine next steps
* `data`: a folder to cache fetched data to improve training/evaluation velocity


### Skills

`/research`: conduct a ML research given a prompt. this serves as the main entry point and it could delegate some of the tasks to other skills.

`/evaluate`: specialized in evaluating ML models.

`/recap`: summarize research findings of recent iterations and distill them into insights and suggestions on next steps

`/reuse`: look across agent-generated scripts / utils in reports and extract reusable scripts for future research. This improves reproducibility and saves token usage in the future.

`/eda`: conduct exploratory data analysis, without training any models.

`/fetch-bars`: a simple skill that serves as a natural language interface to fetch bars from different data sources and summarize basic insights

`/prep-ml-data`: a simple skill that serves as a natural language interface to transform bar data into features and labels and summarize basic insights

`/try-this-idea`: a skill that tries an idea in a given link (could be a published paper or article) by implementing that idea within this framework


### Development Guide

Step 1: clone the repo
```bash
git clone git@github.com:homelessrain/agentic_ml_research_for_trading.git
```

Step 2: Create a Python virtual environment


```bash
python3 -m venv env
source env/bin/activate
```

Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

Step 4: Install Claude

