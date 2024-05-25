# Service Trataments Api

This project is a web application that uses Flask for the backend and Next.js for the frontend. The backend processes text notes to extract specific details and preprocesses the text. The frontend provides an interface to interact with this backend.

## Features
- Text preprocessing: Removes punctuation, converts text to lowercase, and filters out stopwords.
- Detail extraction: Extracts specific information from the text based on predefined patterns.
- CORS support: Allows cross-origin requests from any origin.

## Prerequisites

- Python 3.x
- Node.js
- NPM or Yarn

## Setup

### Backend

**1. Clone the repository:**

```bash

git clone <repository-url>
cd <repository-directory>

```

**2. Set up a virtual environment:**

```bash

python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`

```

**3. Install the required Python packages:**

```bash

pip install -r requirements.txt

```

**4. Set up NLTK data:**

Download the necessary NLTK data files (if not already downloaded). Place the downloaded files in a directory named `nltk_data` within the project root.

```bash

import nltk
nltk.download('punkt', download_dir='./nltk_data')
nltk.download('stopwords', download_dir='./nltk_data')

```

# Running the Application

### Backend

Run the Flask server:

```bash

python app.py

```

The Flask server will be running on `http://127.0.0.1:5000`.

# API Endpoints

### `POST /process`

Processes the provided text notes to extract specific details.

### Request
- Content-Type: application/json
- Body:

```bash

{
  "notes": [
    "Your text note 1",
    "Your text note 2"
  ]
}

```
### Response

- Status: `200 OK`
- Body: JSON array containing the extracted details from each note

```bash

[
  {
    "cause_by": "Detail extracted for cause by",
    "testes_realizados_by": "Detail extracted for tests realizados by",
    "solution_by": "Detail extracted for solution by",
    "validated_by": "Detail extracted for validated by"
  },
  ...
]

```
## Project Structure

```bash

my-nextjs-flask-app/
│
├── api
     ├── app.py         # Flask application
├── requirements.txt    # Python dependencies
├── nltk_data/          # NLTK data files (to be downloaded)
├── package.json        # Dependecies
    └── ...             # Other files

```

### License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request with your improvements.

### Acknowledgements

- [Flask](https://flask.palletsprojects.com/en/latest/)
- [Next.js](https://nextjs.org/)
- [NLTK](https://www.nltk.org/)
- [CORS](https://flask-cors.readthedocs.io/en/latest/)