# Adobe-hackathon-round1b
Adobe India Hackathon - Round 1B

This project delivers a solution for Round 1B of the Adobe India Hackathon 2025, focusing on extracting and prioritizing the most relevant sections from a collection of PDF documents based on a specified persona and job-to-be-done. The solution builds upon the document outlining capabilities developed in Round 1A and adheres strictly to all hackathon constraints, including CPU-only execution, limited model size, and offline operation.

### Approach
Our approach to achieving persona-driven document intelligence involves a multi-stage pipeline that combines robust PDF content extraction, structural analysis from Round 1A, and lightweight semantic relevance ranking.

- **Input Processing:** The solution starts by reading a single ``challenge1b_input.json`` file. This JSON contains the full list of PDF documents to analyze, a detailed ``persona`` definition (role, expertise, focus areas), and the ``job_to_be_done`` (a specific task). The corresponding PDF files are expected to be present in the same input directory.

- Document Outline Extraction (Leveraging Round 1A): For each input PDF, the system seamlessly integrates and calls the ``analyze_pdf_with_ml`` function from the ``round1a_module\outline_extractor``. This component, which uses a custom-trained Logistic Regression model, is responsible for extracting the document's hierarchical outline (Title, H1, H2, H3) along with their respective 0-indexed page numbers. This step provides the foundational structure for identifying meaningful sections.

- Precise Section Content Extraction: After obtaining the outline, the system performs a precise extraction of the content associated with each identified heading. It iterates through all text elements within the PDF, using their bounding box coordinates and page flow. For each heading, it gathers all subsequent text elements on the same page and subsequent pages (until the next heading is encountered or a reasonable page limit is reached). This ensures that the "section content" accurately represents the textual information belonging to that heading, providing rich context for relevance analysis.

- Query Formulation: A comprehensive query string is dynamically constructed by combining all relevant information from the input persona (role, expertise, focus areas) and the ``job_to_be_done`` task. This consolidated query serves as the basis for determining the relevance of document sections.

Semantic Relevance Analysis (TF-IDF & Cosine Similarity):  

- TF-IDF Vectorization: A `TfidfVectorizer` from scikit-learn is used to convert the textual content of each document section and the formulated query into numerical vectors. TF-IDF (Term Frequency-Inverse Document Frequency) assigns weights to words based on how frequently they appear in a given section relative to their frequency across the entire collection of document sections. This highlights terms that are important within a specific section but less common overall.

- Cosine Similarity: The cosine similarity metric is then calculated between the TF-IDF vector of the query and the TF-IDF vector of each extracted document section. Cosine similarity measures the angle between two vectors, with a higher value indicating greater semantic similarity.

- This process is performed on the fly for each document collection, ensuring the vectorizer's vocabulary is tailored to the specific content and eliminating the need for large, pre-trained language models at runtime, thus adhering to the memory and offline constraints.

Ranking and Output Generation:  

- Document sections are ranked in descending order based on their calculated cosine similarity scores to the query. Only sections with a positive relevance score are included.

- The final output is generated in the challenge1b_output.json format, including comprehensive metadata (input documents, persona, job, processing timestamp) and a list of extracted_sections. Each extracted section includes its document filename, 0-indexed page number, section title, importance_rank (1-indexed), and a sub_section_analysis object containing the refined_text (the extracted content) and its page number.

This solution effectively connects structural document understanding with semantic intelligence, providing a powerful tool for targeted information retrieval within the specified hackathon environment.

### Models and Libraries used

- ``pdfminer.six``: For robust PDF text and layout extraction in both outline generation (Round 1A) and content extraction (Round 1B).

- ``numpy``: For numerical operations.

- ``scikit-learn``: For the Machine Learning model (`LogisticRegression` classifier) in Round 1A, and for `TfidfVectorizer` and cosine_similarity in Round 1B.

- ``joblib``: For efficient saving and loading of the trained scikit-learn model and scaler from Round 1A.

- Custom-trained Logistic Regression Model (from Round 1A): A lightweight, locally trained model is used for outline extraction, ensuring compliance with size and offline constraints.

### How to Build and Run
The solution is containerized using Docker, as required by the hackathon.  

Prerequisites:  
- Docker installed and running on your system (Windows, macOS, or Linux).
- Local project directory structures as in the repo.
- Ensure that ``outline_classifier.joblib`` and ``scaler.joblib`` are present in ``round1a_module/model/``. These files are generated by running the ``train_model_colab.py`` script in Google Colab (from Round 1A setup).

**Build Command** (as specified in the problem statement):

Navigate to your adobe-hackathon-round1b directory in your terminal/PowerShell and run:

`docker build --platform linux/amd64 -t mysolutionname:somerandomidentifier .`

Replace mysolutionname:somerandomidentifier with your chosen image name and tag (e.g., adobe-persona-analyzer:v1.0).

It's recommended to use --no-cache flag during development to ensure all changes are picked up:

`docker build --no-cache --platform linux/amd64 -t adobe-persona-analyzer:v1.0 .`

Run Command (as specified in the problem statement):

After building the image, run the solution using:

`docker run --rm -v "$(pwd)/input:/app/input" -v "$(pwd)/output:/app/output" --network none mysolutionname:somerandomidentifier`

Replace mysolutionname:somerandomidentifier with the exact image name and tag used during the build step.

- ``--rm``: Automatically removes the container after it exits.

- ``-v "$(pwd)/input:/app/input"``: Mounts your local input directory (containing PDFs and challenge1b_input.json) to /app/input inside the container.

- ``-v "$(pwd)/output:/app/output"``: Mounts your local output directory to /app/output inside the container.

- ``--network none``: Ensures no network access during execution, fulfilling the offline constraint.

The container will execute persona_analyzer.py, which will process the input documents based on the provided persona and job, and generate ``challenge1b_output.json`` in your local output/ directory.