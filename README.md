# 🇮🇳 SchemeSathi AI

SchemeSathi AI is a Telegram bot that helps people explore Indian Central and State government benefit schemes. Users can build a profile, browse relevant schemes, review structured eligibility checks, and keep a household document-readiness checklist.

🔗 **GitHub:** https://github.com/adithhn12/SchemeSathi-AI  
🤖 **Platform:** Telegram (run your own bot with a Telegram Bot API token)

---

## 🎯 Problem Statement

Information about public benefit schemes is spread across many sources and can be difficult to navigate. People may not know which schemes could fit their circumstances, what conditions are listed, or which documents to prepare.

Common challenges include:

- Finding potentially relevant schemes among a large catalogue
- Understanding listed eligibility requirements
- Comparing benefits, documents, and application information
- Keeping track of documents for different household members
- Distinguishing verified official information from third-party records

---

## 💡 Solution

SchemeSathi provides a conversational Telegram interface to a catalogue of Indian government schemes. It combines profile-based filtering, transparent checks for available structured eligibility fields, text-based recommendations, and document-readiness tools.

The bot is an informational prototype. It does not make official eligibility decisions, verify current scheme rules, or submit applications.

---

## ✨ Features

### 👤 User Profiles

Build and update a profile with details used by the bot to find potentially relevant schemes. Profiles and conversations are stored locally and scoped to the Telegram user.

### 🧭 Scheme Discovery

Browse recommendations, search scheme records, explore life-event topics, and compare scheme details.

### ✅ Structured Eligibility Checks

Where the dataset provides structured fields, the bot can check age, income, state, gender, residence, disability, BPL status, listed social category, and explicit beneficiary occupation conditions. Results distinguish satisfied, failed, and unknown conditions. A missing condition is treated as unknown.

### 📄 Scheme Details and Documents

Review scheme descriptions, benefits, listed requirements, application steps, and dataset-provided links. Document lookup and readiness checklists help organize preparation for household members.

### 🧠 Local Recommendation Tools

TF-IDF and cosine similarity support recommendation and FAQ retrieval. Relevance scores describe text similarity; they are not eligibility probabilities.

### 🗑️ Privacy Controls

Use `/reset` to stop the current conversation flow. Use `/delete_profile` to erase the saved profile, household members, checklist, and conversation state from the local database. The bot does not need Aadhaar numbers, OTPs, passwords, or document numbers.

---

## 🏗️ Architecture

The bot runs as a Python process that polls Telegram, routes messages to conversational features, loads the local scheme catalogue and recommendation model, and stores user data in SQLite.

**Telegram user → Telegram Bot API → SchemeSathi Python bot → CSV catalogue / local ML / SQLite**

---

## 🛠️ Tech Stack

- Python 3.10+
- `python-telegram-bot`
- pandas
- scikit-learn (TF-IDF, Logistic Regression, cosine similarity)
- SQLite
- python-dotenv
- Git and GitHub

---

## 🚀 Getting Started

### 1. Clone the repository

```powershell
git clone https://github.com/adithhn12/SchemeSathi-AI.git
cd SchemeSathi-AI
```

### 2. Create and activate a virtual environment (recommended)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Configure the Telegram bot token

Create a `.env` file in the project root:

```env
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
```

Keep `.env` private. It is excluded from Git.

### 5. Run the bot

```powershell
python schemesathi.py
```

Open your bot in Telegram and send `/start`. The bot needs internet access to communicate with Telegram.

Available commands: `/start`, `/help`, `/reset`, `/about`, `/profile`, and `/delete_profile`.

### Optional: rebuild local reports and model

```powershell
python -m training.build_dataset_report
python -m training.train_models
```

The dataset quality report and model evaluation are written under `reports/`. The trained classifier is written under `models/` and can be regenerated from the included training examples.

---

## 📚 Dataset and Information Limitations

The included `data/Schemes.csv` is sourced from [Sumna/indian-government-schemes-2025 on Hugging Face](https://huggingface.co/datasets/Sumna/indian-government-schemes-2025). The project README describes the supplied dataset as containing 4,693 scheme records. Please refer to the upstream dataset page for its attribution and license terms.

Scheme facts and links in the dataset have not been independently verified against myScheme, data.gov.in, or the responsible government departments. Dataset-provided `official_url` and `apply_url` values are shown as supplied and are not proof of currentness or official verification. Check the responsible department's current instructions before acting.

Eligibility checks only use structured conditions the dataset provides and a limited set of clear facts extracted from conversation. Natural-language descriptions are supporting information, not comprehensive deterministic rules. Recommendation and FAQ scores are text-match signals, not approval likelihoods.

---

## 🔮 Future Enhancements

- Verify scheme details and links against official sources
- Add update timestamps and scheme-currentness checks
- Expand language support
- Improve structured eligibility coverage and user explanations
- Add deployment guidance for a persistent hosted bot
- Add an optional secure cloud storage mode

---

## 👥 Team

Developed as an educational and hackathon project by the SchemeSathi team.

---

## 📄 License

No software license is currently included in this repository. The scheme dataset has its own upstream source and license terms; see the dataset page linked above. Add a repository `LICENSE` file before granting reuse rights for this project's code.
