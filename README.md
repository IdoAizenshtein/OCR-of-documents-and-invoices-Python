# 🔍 Invoice OCR Processing System - Automated Tax & Amount Extraction

[![AWS](https://img.shields.io/badge/AWS-Cloud%20Ready-orange)](https://aws.amazon.com/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-blue)](https://www.docker.com/)
[![Tesseract](https://img.shields.io/badge/Tesseract-OCR-green)](https://github.com/tesseract-ocr/tesseract)
[![Python](https://img.shields.io/badge/Python-3.x-yellow)](https://www.python.org/)

An intelligent, cloud-ready OCR (Optical Character Recognition) system specifically designed for extracting financial data from invoices and documents. The system automatically detects and processes amounts, taxes, and other financial information while correcting document alignment for optimal recognition accuracy.

## 💡 Specialized Features

- **Intelligent Financial Data Extraction**:
  - Automated detection of total amounts
  - Tax calculation and verification
  - Currency symbol recognition
  - Invoice number identification
  
- **Advanced Document Processing**:
  - Automatic skew correction
  - Smart angle alignment
  - Document orientation detection
  - Multi-page document support
  
- **High-Precision OCR Processing**: 
  - Utilizing Google's Tesseract OCR engine
  - Custom training for financial data recognition
  - Pattern matching for amount formats
  - Tax field identification

- **Document Enhancement**:
  - Automatic image straightening
  - Contrast optimization
  - Noise reduction
  - Resolution enhancement

## 🎯 Perfect For

- **Invoice Processing**: Automatically extract and verify invoice totals
- **Tax Documentation**: Identify and validate tax amounts
- **Financial Document Analysis**: Process financial statements and receipts
- **Batch Processing**: Handle multiple documents simultaneously
- **Real-time Processing**: Immediate results through API integration

## 🚀 Key Technical Features

- **Advanced OCR Processing**: Specialized in financial data extraction
- **AWS Integration**: Seamless connection with S3 for storage and EC2 for processing
- **Docker Support**: Containerized deployment for consistent environments
- **Image Pre-processing**: Built-in image cropping and angle correction
- **RESTful API**: Server implementation for easy integration
- **Scalable Architecture**: Cloud-native design for handling large workloads

## 📋 Prerequisites

- Python 3.x
- Docker
- AWS Account
- Tesseract OCR Engine

## 🛠️ Installation

1. Clone the repository:
   bash git clone https://github.com/IdoAizenshtein/OCR-of-documents-and-invoices-Python](https://github.com/IdoAizenshtein/OCR-of-documents-and-invoices-Python.git
2. Install dependencies:
   bash pip install -r requirements.txt
3. Configure AWS credentials:
   bash aws configure
4. Build Docker container:
   bash docker build -t invoice-ocr-processor .

## ⚙️ Configuration

Configure the application using `application.yaml`:
- AWS credentials
- S3 bucket settings
- OCR parameters
- Document processing settings
- Tax recognition patterns
- Amount format configurations

## 🚀 Usage

1. Start the server:
bash python Server.py

2. Process invoices:
```bash
python main.py [document_path]
```
3. For document alignment and cropping:
   python crop.py [document_path]


## 🏗️ Architecture

The system consists of several specialized components:
- Document Alignment Module
- Financial Data Extraction Engine
- Tax Calculation Validator
- Amount Recognition System
- AWS Cloud Integration
- RESTful API Server
- Docker Container

## 🔧 Technical Stack

- **Backend**: Python
- **OCR Engine**: Tesseract
- **Cloud Provider**: AWS (S3, EC2)
- **Containerization**: Docker
- **Image Processing**: Custom Python tools
- **Document Analysis**: Specialized algorithms for financial data

## 💼 Business Applications

- Automated accounting systems
- Financial document processing
- Tax compliance verification
- Invoice validation
- Expense management
- Audit trail documentation

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


## 🔍 Keywords

invoice ocr, tax extraction, amount recognition, document processing, optical character recognition, financial ocr, python ocr, aws ocr, tesseract, docker ocr, cloud ocr, image processing, text extraction, aws s3, aws ec2, python image processing, ocr api, tesseract ocr python, cloud computing, containerized ocr, automated text extraction, document processing, image text recognition, python automation, invoice processing, tax document scanning, financial data extraction, document alignment correction, skew correction

---

<div align="center">
  <img src="Tesseract_OCR_logo_(Google).png" height="60px">
  <img src="Python-logo-notext.svg.png" height="60px">
  <img src="Amazon-S3-Logo.svg" height="60px">
  <img src="AWS_Simple_Icons_Compute_Amazon_EC2_Instances.svg.png" height="60px">
</div>


