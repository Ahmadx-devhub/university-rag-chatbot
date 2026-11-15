from ingest import load_pdfs, split_documents

documents = load_pdfs("data")

chunks = split_documents(documents)

print("\nTotal pages:", len(documents))
print("Total chunks:", len(chunks))

print("\nFirst chunk:")
print(chunks[0].page_content)