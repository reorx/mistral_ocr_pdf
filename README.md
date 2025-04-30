# Mistral OCR PDF

CLI tool to convert PDFs to markdown using Mistral OCR.

## Installation

```bash
pip install mistral-ocr-pdf
# or with uv
uv pip install mistral-ocr-pdf
```

## Usage

Convert a PDF to markdown:

```bash
mistral-ocr-pdf input.pdf
# Output will be saved as input.md
```

Specify output file:

```bash
mistral-ocr-pdf -o output.md input.pdf
```

Save raw OCR response JSON for debugging:

```bash
mistral-ocr-pdf --output-json response.json input.pdf
```

## API Key Configuration

The Mistral API key can be configured in three ways:

1. Environment variable:
   ```bash
   export MISTRAL_API_KEY="sk-xxxx"
   mistral-ocr-pdf input.pdf
   ```

2. Config file:
   ```bash
   mistral-ocr-pdf -sk api_key -sv "sk-xxxx"
   # This saves the key to ~/.config/mistral_ocr_pdf/config.json
   ```

3. Direct config file editing:
   Create or edit `~/.config/mistral_ocr_pdf/config.json` with:
   ```json
   {
     "api_key": "sk-xxxx"
   }
   ```

## Run with uv

You can also run the tool directly with `uvx`:

```bash
uvx mistral-ocr-pdf input.pdf
```