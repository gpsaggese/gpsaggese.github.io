import argparse
import logging
import os
import re
import sys
import time
from typing import List, Tuple

import google.generativeai as genai
from tqdm import tqdm

def configure_logging(verbosity: int) -> None:
    level = logging.INFO
    if verbosity >= 2:
        level = logging.DEBUG
    elif verbosity <= 0:
        level = logging.WARNING

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

def parse_args():
    parser = argparse.ArgumentParser(description="Enhance text file to markdown using Gemini.")
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=1,
        help="Increase log verbosity (-v for INFO, -vv for DEBUG).",
    )
    return parser.parse_args()

def read_file_content(file_path: str) -> str:
    """Reads the entire content of a file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {e}")
        sys.exit(1)

def split_content_into_sections(content: str) -> List[Tuple[int, str]]:
    """Splits the raw content into slides based on slide markers using findall for robustness."""
    sections = []
    
    # This pattern finds all occurrences of slide blocks
    pattern = r'=== SLIDE (\d+) ===\n\n(.*?)\n\n=== END SLIDE ==='
    
    matches = re.findall(pattern, content, flags=re.DOTALL)
    
    for match in matches:
        try:
            slide_number = int(match[0])
            slide_content = match[1].strip()
            
            if slide_content:
                sections.append((slide_number, slide_content))
        except (ValueError, IndexError):
            logging.warning(f"Could not parse a slide block. Match found: {match}")
            
    return sections

def enhance_text_with_gemini(rules: str, section_batch: List[Tuple[int, str]], model, retry_count: int = 0) -> List[str]:
    """Sends a batch of raw slides to Gemini and returns the converted markdown for each slide."""
    batch_content = "\n\n".join([f"Slide {num}:\n{content}" for num, content in section_batch])
    
    separator = "[SLIDE_BREAK]"

    prompt = f"""You are an expert in converting raw slide content into well-structured LaTeX/Pandoc markdown. Your primary task is to replace the raw content markers (e.g., [TITLE], [BULLET:level=0]) with the correct markdown based on the instructions below. Use the provided JSON rules as a stylistic guide for the final output.

**CRITICAL INSTRUCTIONS:**
1. You MUST process ALL {len(section_batch)} slides provided.
2. You MUST place "{separator}" between each slide's converted content.
3. Convert all raw markers to markdown. DO NOT leave any raw markers like `[BULLET:...]` in the final output.
4. Follow the JSON rules for stylistic formatting.
5. Ensure all output is LaTeX compatible.
6. DO NOT use dollar signs ($) unless for math expressions specified in the rules.
7. Each slide should have a unique title/heading.
8. Add "* {{heading_name}}" after each major heading.

**Raw Content Marker Conversion Guide (Examples):**
- `[TITLE]\nSome Title` → ` # ##############################################################################\n# Some Title\n# ##############################################################################\n\n* Some Title`
- `[SUBTITLE]\nSome Subtitle` → `## Some Subtitle`
- `[TEXT]\nSome text.` → `Some text.`
- `[BULLET:level=0] Item 1` → `- Item 1`
- `[BULLET:level=1] Sub-item A` → `  - Sub-item A`
- `[BOLD]text[/BOLD]` → `**text**`
- `[ITALIC]text[/ITALIC]` → `*text*`
- `[IMAGE] path/to/image.png` → `![](path/to/image.png)`
- `[TABLE 4x2]` with `[CELL:0,0|HEADER]...` lines → A full markdown table.
- DO NOT NEED SPEAKER NOTES

**Stylistic Rules (from JSON):**
```json
{rules}
```

**Raw Slide Content ({len(section_batch)} slides total):**
```text
{batch_content}
```

IMPORTANT: Your response must contain exactly {len(section_batch)} converted slides separated by "{separator}". The output must be clean markdown with no raw markers remaining. Do not include any other text, explanations, or comments."""

    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            time.sleep(1)  # To avoid hitting rate limits
            
            if not response.text or not response.text.strip():
                logging.warning(f"Empty response from Gemini API on attempt {attempt + 1}")
                continue
            
            # Split the response into individual slides using the unique separator
            enhanced_slides = response.text.strip().split(separator)
            
            # Filter out any empty strings that might result from the split
            enhanced_slides = [slide.strip() for slide in enhanced_slides if slide.strip()]
            
            logging.info(f"Batch processing attempt {attempt + 1}: Expected {len(section_batch)} slides, got {len(enhanced_slides)}")
            
            # Check if we got the expected number of slides
            if len(enhanced_slides) == len(section_batch):
                logging.info(f"Successfully processed batch of {len(section_batch)} slides")
                return enhanced_slides
            
            # If we got close (within 1), try to fix it
            elif abs(len(enhanced_slides) - len(section_batch)) <= 1:
                logging.warning(f"Slide count mismatch but close enough. Expected {len(section_batch)}, got {len(enhanced_slides)}")
                
                # Pad with empty sections if we got fewer
                while len(enhanced_slides) < len(section_batch):
                    enhanced_slides.append("# Content processing error - slide missing")
                
                # Trim if we got more
                enhanced_slides = enhanced_slides[:len(section_batch)]
                
                return enhanced_slides
            
            else:
                logging.warning(f"Significant slide count mismatch on attempt {attempt + 1}. Expected {len(section_batch)}, got {len(enhanced_slides)}")
                if attempt == max_retries - 1:
                    logging.error("Max retries reached for batch processing")
                    break
                continue
                
        except Exception as e:
            logging.error(f"Error calling Gemini API on attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                logging.error("Max retries reached due to API errors")
                break
            time.sleep(2)  # Wait longer before retry
            continue
    
    # If all retries failed, try to split the batch in half (but maintain minimum batch size)
    if len(section_batch) >= 10:  # Only split if we have enough sections
        logging.warning(f"Splitting batch of {len(section_batch)} slides into smaller batches")
        mid_point = len(section_batch) // 2
        first_half = section_batch[:mid_point]
        second_half = section_batch[mid_point:]
        
        first_results = enhance_text_with_gemini(rules, first_half, model, retry_count + 1)
        second_results = enhance_text_with_gemini(rules, second_half, model, retry_count + 1)
        
        return first_results + second_results
    
    # Final fallback: return placeholder content to maintain structure
    logging.error(f"Failed to process batch of {len(section_batch)} slides after all retries")
    placeholder_results = []
    for num, content in section_batch:
        # Return the original content with a warning header
        placeholder_results.append(f"# PROCESSING ERROR - Slide {num}\n{content}")
    
    return placeholder_results


def main():
    args = parse_args()
    configure_logging(args.verbose)

    logging.info("Starting enhancement process...")

    rules_file_path = "rules.txt"
    
    # Find the most recent raw file in output directory
    output_dir = "output"
    if not os.path.exists(output_dir):
        logging.error("Output directory not found.")
        logging.error("Please run slides_to_text.py first to generate the raw content.")
        sys.exit(1)
        
    raw_files = [f for f in os.listdir(output_dir) if f.endswith("_raw.txt")]
    
    if not raw_files:
        logging.error("No raw text files found in output directory.")
        logging.error("Please run slides_to_text.py first to generate the raw content.")
        sys.exit(1)
    
    # Use the most recent file
    raw_files.sort(key=lambda x: os.path.getmtime(os.path.join(output_dir, x)), reverse=True)
    input_file_path = os.path.join(output_dir, raw_files[0])
    output_file_path = os.path.join(output_dir, "final_enhanced_markdown.txt")
    
    logging.info(f"Processing raw file: {input_file_path}")

    rules = read_file_content(rules_file_path)
    content = read_file_content(input_file_path)
    
    sections = split_content_into_sections(content)
    logging.info(f"Found {len(sections)} slides to process.")

    # Configure the Gemini client
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logging.warning("Set GOOGLE_API_KEY in environment variable to use Gemini API.")
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        logging.info("Gemini client configured successfully.")
    except Exception as e:
        logging.error(f"Failed to configure Gemini client: {e}")
        logging.error("Please make sure the GOOGLE_API_KEY environment variable is set correctly.")
        sys.exit(1)
    
    # Force minimum batch size of 5, preferred batch size of 8
    min_batch_size = 5
    preferred_batch_size = 8
    
    enhanced_content = []
    
    # Create batches ensuring minimum size
    section_batches = []
    for i in range(0, len(sections), preferred_batch_size):
        batch = sections[i:i + preferred_batch_size]
        
        # If this is the last batch and it's smaller than minimum, merge with previous
        if len(batch) < min_batch_size and section_batches:
            section_batches[-1].extend(batch)
        else:
            section_batches.append(batch)
    
    # Ensure we have proper batches
    if not section_batches and sections:
        section_batches = [sections]  # Process all as one batch if very few sections
    
    logging.info(f"Created {len(section_batches)} batches with sizes: {[len(batch) for batch in section_batches]}")

    for i, batch in enumerate(tqdm(section_batches, desc="Converting slides to markdown")):
        batch_numbers = [num for num, _ in batch]
        logging.info(f"Processing batch {i+1}/{len(section_batches)} with {len(batch)} slides: {batch_numbers}")
        
        enhanced_batch = enhance_text_with_gemini(rules, batch, model)
        enhanced_content.extend(enhanced_batch)
        
        # Add a small delay between batches to be respectful to the API
        if i < len(section_batches) - 1:
            time.sleep(2)

    try:
        with open(output_file_path, "w", encoding="utf-8") as f:
            # Join content with appropriate spacing
            f.write("\n\n".join(enhanced_content))
        logging.info(f"Successfully wrote enhanced content to {output_file_path}")
        logging.info(f"Converted {len(enhanced_content)} slides total")
    except Exception as e:
        logging.error(f"Error writing to output file {output_file_path}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 