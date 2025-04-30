#!/usr/bin/env python3
"""
Mistral OCR PDF - Convert PDFs to markdown using Mistral OCR API
"""
import os
import sys
import json
import base64
import argparse
from pathlib import Path
from mistralai import Mistral, DocumentURLChunk
from mistralai.models import OCRResponse


__version__ = "0.1.0"


def get_api_key():
    api_key = os.environ.get("MISTRAL_API_KEY")
    if api_key:
        return api_key

    config_path = Path.home() / ".config" / "mistral_ocr_pdf" / "config.json"
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
                api_key = config.get("api_key")
                if api_key:
                    return api_key
        except Exception as e:
            print(f"Error reading config file: {e}", file=sys.stderr)

    return None


def set_config_value(key, value):
    config_dir = Path.home() / ".config" / "mistral_ocr_pdf"
    config_path = config_dir / "config.json"
    
    # Ensure directory exists
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Load existing config or create new one
    config = {}
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
        except Exception as e:
            print(f"Warning: Could not read existing config: {e}", file=sys.stderr)
    
    # Update config
    config[key] = value
    
    # Write back to file
    try:
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Successfully set {key} in {config_path}")
        return True
    except Exception as e:
        print(f"Error writing config file: {e}", file=sys.stderr)
        return False


def replace_images_in_markdown_with_wikilinks(markdown_str: str, image_mapping: dict) -> str:
    updated_markdown = markdown_str
    for original_id, new_name in image_mapping.items():
        updated_markdown = updated_markdown.replace(
            f"![{original_id}]({original_id})",
            f"![[{new_name}]]"
        )
    return updated_markdown


def process_pdf(pdf_path: Path, api_key: str, output_json_path: Path = None) -> str:
    """
    Process a PDF file using Mistral OCR and return markdown content.
    
    Args:
        pdf_path: Path to the PDF file
        api_key: Mistral API key
        output_json_path: Optional path to save raw OCR response
        
    Returns:
        The markdown content
    """
    client = Mistral(api_key=api_key)
    uploaded_file = None
    pdf_base = pdf_path.stem
    
    try:
        print(f"Processing {pdf_path.name}...")
        
        # Read and upload the PDF
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        
        uploaded_file = client.files.upload(
            file={"file_name": pdf_path.name, "content": pdf_bytes}, purpose="ocr"
        )
        
        print(f"File uploaded (ID: {uploaded_file.id}). Getting signed URL...")
        signed_url = client.files.get_signed_url(file_id=uploaded_file.id, expiry=60)
        
        print(f"Calling OCR API...")
        ocr_response = client.ocr.process(
            document=DocumentURLChunk(document_url=signed_url.url),
            model="mistral-ocr-latest",
            include_image_base64=True
        )
        print(f"OCR processing complete for {pdf_path.name}.")
        
        # Save raw OCR response if requested
        if output_json_path:
            try:
                with open(output_json_path, "w", encoding="utf-8") as json_file:
                    if hasattr(ocr_response, 'model_dump'):
                        json.dump(ocr_response.model_dump(), json_file, indent=4, ensure_ascii=False)
                    else:
                        json.dump(ocr_response.dict(), json_file, indent=4, ensure_ascii=False)
                print(f"Raw OCR response saved to {output_json_path}")
            except Exception as json_err:
                print(f"Warning: Could not save raw OCR JSON: {json_err}", file=sys.stderr)
        
        # Process OCR Response -> Markdown & Images
        global_image_counter = 1
        updated_markdown_pages = []
        
        # Create images directory alongside output markdown file if needed
        images_dir = None
        
        for page_index, page in enumerate(ocr_response.pages):
            current_page_markdown = page.markdown
            page_image_mapping = {}
            
            # Extract images if there are any
            if page.images and not images_dir:
                images_dir = pdf_path.parent / f"{pdf_base}_images"
                images_dir.mkdir(exist_ok=True)
            
            for image_obj in page.images:
                base64_str = image_obj.image_base64
                if not base64_str:
                    continue
                
                if base64_str.startswith("data:"):
                    try:
                        base64_str = base64_str.split(",", 1)[1]
                    except IndexError:
                        continue
                
                try:
                    image_bytes = base64.b64decode(base64_str)
                except Exception as decode_err:
                    print(f"Warning: Base64 decode error for image {image_obj.id} on page {page_index+1}: {decode_err}", file=sys.stderr)
                    continue
                
                original_ext = Path(image_obj.id).suffix
                ext = original_ext if original_ext else ".png"
                new_image_name = f"{pdf_base}_p{page_index+1}_img{global_image_counter}{ext}"
                global_image_counter += 1
                
                image_output_path = images_dir / new_image_name
                try:
                    with open(image_output_path, "wb") as img_file:
                        img_file.write(image_bytes)
                    page_image_mapping[image_obj.id] = new_image_name
                except IOError as io_err:
                    print(f"Warning: Could not write image file {image_output_path}: {io_err}", file=sys.stderr)
                    continue
            
            updated_page_markdown = replace_images_in_markdown_with_wikilinks(current_page_markdown, page_image_mapping)
            updated_markdown_pages.append(updated_page_markdown)
        
        # Clean up Mistral file
        try:
            client.files.delete(file_id=uploaded_file.id)
            print(f"Deleted temporary file {uploaded_file.id} from Mistral.")
        except Exception as delete_err:
            print(f"Warning: Could not delete file {uploaded_file.id} from Mistral: {delete_err}", file=sys.stderr)
        
        return "\n\n---\n\n".join(updated_markdown_pages)
        
    except Exception as e:
        error_str = str(e)
        # Attempt to extract JSON error message from the exception string
        json_index = error_str.find('{')
        if json_index != -1:
            try:
                error_json = json.loads(error_str[json_index:])
                error_msg = error_json.get("message", error_str)
            except Exception:
                error_msg = error_str
        else:
            error_msg = error_str
        
        print(f"Error processing {pdf_path.name}: {error_msg}", file=sys.stderr)
        
        # Attempt cleanup even on error
        if uploaded_file:
            try:
                client.files.delete(file_id=uploaded_file.id)
            except Exception:
                pass
        
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Convert PDF files to markdown using Mistral OCR")
    
    # Config options
    parser.add_argument("-sk", "--set-key", help="Config key to set")
    parser.add_argument("-sv", "--set-value", help="Value to set for the config key")
    
    # Input and output file options
    parser.add_argument("input", help="Input PDF file", nargs="?")
    parser.add_argument("-o", "--output", help="Output markdown file (default: input filename with .md extension)")
    parser.add_argument("--output-json", help="Save raw OCR response to a JSON file")
    
    args = parser.parse_args()
    
    # Handle setting config values
    if args.set_key and args.set_value:
        if set_config_value(args.set_key, args.set_value):
            return 0
        else:
            return 1
            
    # Ensure we have an input file when not setting config
    if not args.input:
        if args.set_key or args.set_value:
            parser.error("--set-key and --set-value must be used together")
        else:
            parser.error("input file is required when not setting config values")
    
    # Process PDF file
    pdf_path = Path(args.input)
    if not pdf_path.exists():
        print(f"Error: Input file {pdf_path} does not exist", file=sys.stderr)
        return 1
    
    if not pdf_path.suffix.lower() == '.pdf':
        print(f"Error: Input file must be a PDF file", file=sys.stderr)
        return 1
    
    # Get API key
    api_key = get_api_key()
    if not api_key:
        print("Error: Mistral API key not found. Set it with MISTRAL_API_KEY environment variable "
              "or use -sk api_key -sv your-api-key to set it in the config file", file=sys.stderr)
        return 1
    
    # Determine output path
    output_path = None
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = pdf_path.with_suffix('.md')
    
    # Process the PDF
    markdown_content = process_pdf(
        pdf_path=pdf_path,
        api_key=api_key,
        output_json_path=Path(args.output_json) if args.output_json else None
    )
    
    # Write markdown to file
    try:
        with open(output_path, "w", encoding="utf-8") as md_file:
            md_file.write(markdown_content)
        print(f"Markdown generated successfully at {output_path}")
        return 0
    except IOError as io_err:
        print(f"Error: Failed to write markdown file: {io_err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())