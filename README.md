# Customer Clustering

Project is the solution for eCommerce business assignment.
Having been given users, sessions, deliveries and products frames, we have to deal with the following business problem:

> We do have additional benefits for our best customers, but maybe it would be possible to find out who is potentially willing to spend more with us?

Project covers:
* Explanatory data analysis
* Customer segmentation models proposals
* ML Pipelines
* Microservice deployment

More information was reported in the polish language [here](https://github.com/hpiotr6/Customers-Clustering/blob/main/notebooks/etap2/etap2.ipynb).

* **Python libraries used:** pandas, sklearn, seaborn, log, click, pickle, numpy, fastapi, pydantic, pytest
* **Input:** users, sessions, deliveries and products frames
* **Output:** best clients group


## Installation

1. Install `poetry`: https://python-poetry.org/docs/#installation
2. Create an environment with `poetry install`
3. Run `poetry shell`
4. To run unit tests for your service use `poetry run pytest` or simply `pytest` within `poetry shell`.

<p><small>Project partially based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>


## Usage

```bash
# Running microservice
cd ium-21z/microservice
uvicorn main:app --reload
```

## License
[MIT](https://choosealicense.com/licenses/mit/)



