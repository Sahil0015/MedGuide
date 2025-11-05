import os
import fitz  # PyMuPDF

def convert_all_pdfs_to_txt():
    """
    Convert all PDF files in ../data/knowledge_base_pdfs to text files
    and save them in ../data/knowledge_base
    """

    # Define paths relative to this script
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_folder = os.path.join(base_dir, "data", "knowledge_base_pdfs")
    output_folder = os.path.join(base_dir, "data", "knowledge_base")

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Get list of PDF files
    pdf_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print("⚠️ No PDF files found in:", input_folder)
        return

    for pdf_file in pdf_files:
        pdf_path = os.path.join(input_folder, pdf_file)
        base_name = os.path.splitext(pdf_file)[0]
        output_path = os.path.join(output_folder, f"{base_name}.txt")

        print(f"🔹 Extracting: {pdf_file} ...")

        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page_num, page in enumerate(doc, start=1):
                text += f"\n--- Page {page_num} ---\n"
                text += page.get_text("text")
            doc.close()

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)

            print(f"✅ Saved: {output_path}")

        except Exception as e:
            print(f"❌ Error processing {pdf_file}: {e}")

    print("\n🎉 All PDFs have been converted successfully!")

# Run directly
if __name__ == "__main__":
    convert_all_pdfs_to_txt()
