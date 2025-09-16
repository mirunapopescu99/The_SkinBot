# The SkinBot

Student ID: 24039973

Student Name: Miruna Popescu 

SkinBot is a conversational AI chatbot designed to provide safe, budget-aware skincare recommendations. Unlike generic LLMs, SkinBot is grounded in curated data through combining a product catalog, a system rubric for cautious responses, and fallback conversational seeds — to improve user trust, safety, and willingness to act on skincare guidance.

## Features

- Conversational AI — built on a hosted Large Language Model (LLM).

- Cautious, evidence-based tone — avoids medical claims, always encourages professional validation.

- Grounded responses — linked to a curated skincare product catalog (brand, name, price, retailer, URL).

- Interactive interface — powered by Gradio

- Budget-awareness — recommends affordable products (e.g., “under £15”).

- Disclaimers for sensitive scenarios — e.g., pregnancy, acne, or allergies.


 ## How to Run (Colab)

SkinBot is designed to run in Google Colab with minimal setup.

1. Open Colab

Go to Google Colab
 and start a new notebook.

2. Install dependencies

```
%pip -q install --upgrade "gradio>=4.44.0" "pandas>=2.2.2" "requests>=2.31.0"
```

3. Clone/download this repo

```
!git clone https://github.com/YOUR-USERNAME/skinbot.git
%cd skinbot
```

4. Mount Google Drive

Upload your skincare catalog CSV (skincare_products_clean.csv) into your Google Drive.

from google.colab import drive
```
drive.mount('/content/drive', force_remount=True)
```

SkinBot will look for:
```
/content/drive/MyDrive/skincare_products_clean.csv
```

5. Run the chatbot
```
!python unsloth/chat_transformers_adapter.py
```

Gradio will launch and give you a link, e.g.:
```
http://127.0.0.1:7860  (local)
https://xxxxx.gradio.live (shareable)
```

## API Keys

SkinBot uses OpenRouter
 (default) or OpenAI.

In Colab, set your key before running the script:
```
import os
os.environ["OPENROUTER_API_KEY"] = "your_api_key_here"
```

## User Study

A small-scale user study (N=6) assessed trust, safety, reliance, and long-term use.
Findings:

Trust was conditional: users cross-checked advice with Google/social media/dermatologists.

Safety disclaimers increased trust, but consistency was critical.

Budget-aware routines were appreciated, but users wanted more transparency in sources.

Future Work

Persistent safety disclaimers in every response.

More diverse catalog entries.

Personalisation by skin type & condition.

Streamlit/web deployment for easier public access.

Developed as part of a Master’s dissertation at UAL: Creative Computing Institute (2025).
Data Science and AI in The Creative Industries 
