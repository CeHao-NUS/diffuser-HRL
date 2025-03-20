from scholarly import scholarly

query = "deep learning in robotics"
search_results = scholarly.search_pubs(query)

with open("papers.bib", "w") as bibfile:
    for result in search_results:
        # Construct the BibTeX entry if it doesn't exist as a string
        bib_info = result["bib"]  # Access the 'bib' dictionary
        if isinstance(bib_info, dict):
            # Example BibTeX entry construction
            bibtex_entry = (
                f"@article{{{bib_info.get('title', 'unknown').replace(' ', '_')},\n"
                f"  author = {{{bib_info.get('author', 'unknown')}}},\n"
                f"  title = {{{bib_info.get('title', 'unknown')}}},\n"
                f"  journal = {{{bib_info.get('journal', 'unknown')}}},\n"
                f"  year = {{{bib_info.get('year', 'unknown')}}}\n"
                f"}}\n\n"
            )
            bibfile.write(bibtex_entry)
        else:
            print(f"Unexpected format for: {result}")
