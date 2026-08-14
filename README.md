# AI-Powered University Poster Summarization and Student Notification System

> A university-focused Generative AI application that converts unstructured event posters into structured information, retrieves relevant information using RAG, identifies suitable students, generates personalized emails, creates a supporting image locally, and sends the approved communication through Gmail SMTP.

**Submitted by:** Leo Samuel Gilbert  
**Registration Number:** 2582402

---

## 📌 Overview

Universities regularly publish conferences, workshops, seminars, competitions, academic opportunities, and other notices as posters or documents. Students may miss relevant opportunities because notices are often distributed broadly rather than being matched to their academic profiles.

This project provides an end-to-end workflow that processes a university poster, extracts reliable information, answers questions about the notice, summarizes the event, identifies relevant students, generates personalized communication, creates a supporting image, and sends the communication after user review and confirmation.

The application is implemented as a Streamlit-based interface and uses local AI components for the core language, vision, and image-generation workflow.

---

## 🎯 Problem Statement

University notices are commonly shared as posters containing information such as event titles, dates, deadlines, venues, eligibility, registration details, speakers, and contact information.

Processing these notices manually requires the user to:

1. Read and interpret the poster.
2. Extract important event information.
3. Identify relevant dates and deadlines.
4. Answer questions about the notice.
5. Summarize the event.
6. Identify students for whom the event is relevant.
7. Prepare personalized communication.
8. Create supporting visual content.
9. Review the generated communication.
10. Send the approved communication.

This project combines these stages into a single AI-powered workflow.

---

## 🚀 Key Features

### 1. Poster / Notice Processing

- Upload university event posters through the Streamlit interface.
- Process poster images using OCR.
- Preserve the original poster for optional email attachment.
- Extract textual information from the uploaded notice.

### 2. Vision Language Model Analysis

- Analyse the poster using a local vision-capable model.
- Use visual information to understand poster layout, hierarchy, and relationships between text elements.
- Combine visual information with OCR output during metadata extraction.
- Treat OCR as the primary source for exact textual information.

### 3. Structured Metadata Extraction

Poster information is converted into structured metadata containing fields such as:

- Poster type
- Title
- Department
- University
- Description
- Short summary
- Event date
- Important dates
- Venue
- Registration deadline
- Registration link
- Guest speakers
- Contact person
- Contact phone
- Email
- Eligibility
- Target audience
- Keywords
- Other information

The extraction process uses source-grounded prompting to reduce unsupported information generation.

Target audience classification uses:

- `Explicit`
- `Inferred`
- `Default`

### 4. Poster Summarization

- Generate a concise summary from the extracted poster information.
- Use the summary as additional context for later personalized communication.
- Keep the summary grounded in the poster content.

### 5. Retrieval-Augmented Generation (RAG)

The application provides question answering over the uploaded poster.

Example questions:

```text
What is this event about?

What are the important dates and deadlines?

Where is the event being conducted?

Who are the speakers?

What is the registration deadline?
```

The RAG workflow retrieves relevant poster information and provides answers based on the processed notice.

### 6. Student Matching

- Upload student information through an Excel file.
- Compare student information with the extracted poster metadata.
- Identify students relevant to the event.
- Use information such as programme, department, level, and interests for matching.
- Display student matching information through the application.

### 7. Personalized Email Generation

The email-generation stage combines:

```text
Student Profile
       +
Poster Metadata
       +
Poster Summary
       +
RAG Context
       ↓
Personalized Email Draft
```

Each generated communication can contain:

- Recipient
- Subject
- Personalized email body
- Event-specific information

Generated emails are presented for review before delivery.

### 8. Local AI Image Generation

- Generate a supporting image related to the event.
- Use local Stable Diffusion for image generation.
- Associate the generated image with the personalized communication.
- Display the generated image during the email review stage.

### 9. Email Review and Confirmation

Before sending, the application displays the communication for review.

The review stage includes:

- Recipient
- Subject
- Personalized email body
- Generated supporting image
- Original poster attachment

The communication proceeds to delivery only after confirmation.

### 10. Gmail SMTP Delivery

- Send confirmed personalized emails using Gmail SMTP.
- Include the generated supporting image.
- Optionally attach the original university poster.
- Display the sending result through the application.

---

## 🏗️ System Architecture

```text
                    University Poster
                           |
                           v
                    OCR / Text Extraction
                           |
                           v
                 Vision Language Analysis
                           |
                           v
                  Metadata Extraction
                           |
                           +------------------+
                           |                  |
                           v                  v
                    Poster Summary      RAG Question
                                             Answering
                           |                  |
                           +--------+---------+
                                    |
                                    v
                           Student Information
                                    |
                                    v
                           Student Matching
                                    |
                                    v
                    Personalized Email Generation
                                    |
                                    v
                         Local Image Generation
                                    |
                                    v
                       Complete Email Review
                                    |
                                    v
                          User Confirmation
                                    |
                                    v
                             Gmail SMTP
                                    |
                                    v
                         Email Delivery Status
```

### Critical Workflow Order

```text
Poster
  ↓
OCR + Vision
  ↓
Metadata
  ↓
Summary + RAG
  ↓
Student Matching
  ↓
Personalized Email
  ↓
Local AI Image
  ↓
Complete Communication Review
  ↓
User Confirmation
  ↓
Gmail SMTP
```

---

## 🧩 Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Web Application | Streamlit |
| OCR | EasyOCR |
| Vision / Multimodal Analysis | Gemma 3 |
| Local LLM | Gemma 3 through Ollama |
| Metadata Extraction | Local LLM + Structured JSON |
| RAG | LangChain + Chroma |
| Embeddings | Ollama Embeddings |
| Vector Database | Chroma |
| Student Data | Excel |
| Data Processing | Pandas |
| Image Generation | Stable Diffusion |
| Email Generation | Local LLM |
| Email Delivery | Gmail SMTP |

---

## 🧠 Main Components

### Poster Upload and Processing

The application accepts a university event poster through the Streamlit interface. The uploaded poster is processed for textual and visual information.

### OCR Processing

OCR extracts information from the poster, including:

- Event title
- University
- Department
- Dates
- Venue
- Registration information
- Contact information
- Eligibility
- Other event details

OCR is treated as the primary textual source for exact information.

### Vision Language Model

The Vision Language Model provides supplementary information about:

- Poster structure
- Visual hierarchy
- Relationships between text elements
- Information that may not have been captured correctly by OCR

The OCR and VLM outputs are combined during metadata extraction.

### Structured Metadata Extraction

Gemma 3 processes the OCR text and VLM output and produces structured metadata.

The metadata structure includes:

```json
{
    "poster_type": "...",
    "title": "...",
    "department": "...",
    "university": "...",
    "description": "...",
    "short_summary": "...",
    "event_date": "...",
    "important_dates": [],
    "venue": "...",
    "registration_deadline": "...",
    "registration_link": "...",
    "guest_speakers": [],
    "contact_person": "...",
    "contact_phone": [],
    "email": "...",
    "eligibility": "...",
    "target_audience": "...",
    "target_audience_type": "...",
    "keywords": [],
    "other_information": "..."
}
```

The extraction prompt uses source-grounding rules so that factual information such as names, dates, phone numbers, email addresses, URLs, speakers, eligibility, and event details are extracted only when supported by the available poster information.

### Poster Summarization

The extracted poster information is converted into a concise summary containing the main event information and purpose.

### Retrieval-Augmented Generation

Poster information is stored and retrieved through the RAG pipeline.

The RAG stage allows the user to ask questions about the uploaded poster and retrieve relevant information from the processed content.

### Student Matching

Student information is uploaded through Excel and compared with the event information.

The matching process uses information such as:

- Programme
- Department
- Level
- Interests

The target audience extracted from the poster is also used as part of the event information available for student matching.

### Personalized Email Generation

For relevant students, the local language model generates personalized email drafts using:

```text
Student Profile
       +
Poster Metadata
       +
Poster Summary
       +
RAG Context
```

### Local AI Image Generation

The image-generation stage follows personalized email generation:

```text
Personalized Email
       ↓
Event Context
       ↓
Stable Diffusion
       ↓
Supporting Image
       ↓
Complete Email Review
```

The generated image is displayed with the personalized communication before confirmation.

### Gmail SMTP

The final communication is sent through Gmail SMTP after confirmation.

The email can contain:

```text
Recipient
Subject
Personalized Body
Generated Supporting Image
Original Poster Attachment
```

---

## 📁 Repository Structure

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── app.py
├── src/
├── tests/
└── docs/
    ├── screenshots/
    │   ├── home_page.png
    │   ├── upload_student_excel.png
    │   ├── upload_poster_with_credentials.png
    │   ├── Poster_metadata.png
    │   ├── rag_question.png
    │   ├── rag-question_answer.png
    │   ├── email_generator_1.png
    │   ├── generated_image.png
    │   ├── mail_attachment_confirmation.png
    │   ├── review_email_drafts.png
    │   ├── send_mail.png
    │   ├── mail_send_confirmation.png
    │   └── mail_gmail_received.png
    ├── architecture/
    │   └── architecture.png
    ├── demo/
    │   └── demo.mp4
    └── data/
```

The `docs/screenshots/` directory contains the application interface screenshots used in this documentation.

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd <repository-directory>
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / macOS

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🦙 Ollama Setup

The local language-model workflow uses **Gemma 3 through Ollama**.

Pull the model:

```bash
ollama pull gemma3:latest
```

Verify the available models:

```bash
ollama list
```

Start Ollama:

```bash
ollama serve
```

The application uses the locally running Ollama service for the language-model stages.

---

## 🎨 Local Image Generation

The application uses a local Stable Diffusion workflow for generating the supporting image associated with the personalized email.

The image-generation stage occurs after personalized email generation and before the final communication review.

---

## ▶️ Running the Application

Start the Streamlit application from the project root:

```bash
streamlit run app.py
```

The application provides the interface for:

```text
Poster Upload
      ↓
OCR + Vision
      ↓
Metadata Extraction
      ↓
RAG Question Answering
      ↓
Poster Summary
      ↓
Student Matching
      ↓
Personalized Email Generation
      ↓
AI Image Generation
      ↓
Email Review
      ↓
Email Confirmation
      ↓
Gmail SMTP Delivery
```

---

## 🔄 End-to-End Processing

```text
University Poster
        ↓
OCR Extraction
        ↓
Vision Analysis
        ↓
Structured Metadata
        ↓
Poster Summary
        ↓
RAG Question Answering
        ↓
Student Information
        ↓
Student Matching
        ↓
Personalized Email
        ↓
AI-Generated Image
        ↓
Email Review
        ↓
Confirmation
        ↓
Gmail SMTP
        ↓
Delivered Email
```

### Application Flow

1. Open the Streamlit application.
2. Enter the sender Gmail credentials.
3. Upload the university poster.
4. Upload the student Excel file.
5. Enter questions for the RAG stage.
6. Process the poster.
7. Review the extracted metadata.
8. Review the RAG responses and poster summary.
9. Review the matched students.
10. Generate personalized email drafts.
11. Generate the supporting AI image.
12. Review the complete email and attachments.
13. Confirm the communication.
14. Send the confirmed email through Gmail SMTP.

---

## 📸 Application Screenshots

### Screenshots

## Screenshots

### 1. Home Page

![Home Page](home_page.png)

### 2. Upload Poster and Credentials

![Upload Poster Interface](upload_poster_interface_with_credentials.png)

### 3. Upload Student Excel File

![Upload Student Excel](upload_student_excel.png)

### 4. Extracted Poster Metadata

![Poster Metadata](Poster_metadata.png)

### 5. RAG Question

![RAG Question](rag_question.png)

### 6. RAG Question Answer

![RAG Question Answer](rag_question_answer.png)

### 7. Generated Personalized Email

![Email Generator](email_generator_1.png)

### 8. Review Email Drafts

![Review Email Drafts](review_email_drafts.png)

### 9. Generated AI Image

![Generated AI Image](generated_image.png)

### 10. Email Review and Attachment Confirmation

![Mail Attachment Confirmation](mail_attachment_confirmation.png)

### 11. Send Mail Confirmation

![Send Mail Confirmation](mail_send_confirmation.png)

### 12. Gmail Received Email

![Gmail Received Email](mail_gmail_received.png)

---

## 🔐 Security

The application uses Gmail credentials for email delivery and processes student information.

Credentials and private student information are kept outside the source code and repository.

```text
.env
Private student data
Gmail credentials
Uploaded private posters
Temporary files
Python cache files
```

The Gmail App Password is used for SMTP authentication rather than storing a regular Gmail account password in the application.

---

## 📊 Project Output

The complete workflow produces:

- Structured poster metadata
- Poster summary
- Searchable poster information
- RAG question-answering responses
- Relevant student selection
- Personalized email drafts
- AI-generated supporting image
- Reviewed email with attachments
- Gmail delivery status

The overall transformation is:

```text
Unstructured University Poster
              ↓
       Structured Metadata
              ↓
        RAG + Summary
              ↓
       Relevant Students
              ↓
     Personalized Email
              ↓
       AI-Generated Image
              ↓
        Human Review
              ↓
     Confirmed Communication
              ↓
         Gmail Delivery
```

---

## 📚 Project Objectives

1. Extract reliable information from university poster images.
2. Combine OCR with visual language understanding.
3. Convert poster information into structured metadata.
4. Provide question answering over poster content using RAG.
5. Generate concise poster summaries.
6. Identify students relevant to individual events.
7. Generate personalized student communication.
8. Generate supporting visual content using local image generation.
9. Provide an email review and confirmation stage.
10. Deliver approved communication through Gmail SMTP.

---

## 👤 Author

**Leo Samuel Gilbert**  
**Registration Number:** 2582402

---

## ⭐ Project Summary

> **AI-Powered University Poster Summarization and Student Notification System** transforms unstructured university posters into structured and searchable information, answers poster-related questions using RAG, identifies relevant students, generates personalized communication, creates supporting visuals locally, and delivers the approved communication through Gmail SMTP.
