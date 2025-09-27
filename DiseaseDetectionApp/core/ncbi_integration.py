# DiseaseDetectionApp/core/ncbi_integration.py
import requests
import xml.etree.ElementTree as ET

def get_pubmed_summary(disease_name, max_results=3):
    """
    Fetches recent research summaries for a given disease from the PubMed database.

    Args:
        disease_name (str): The name of the disease to search for.
        max_results (int): The maximum number of article summaries to retrieve.

    Returns:
        str: A formatted string containing summaries of recent research, or a message
             indicating that no results were found or an error occurred.
    """
    print(f"Searching PubMed for articles related to: {disease_name}")
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    search_term = f"{disease_name.replace(' ', '+')}[Title/Abstract]"

    # Step 1: Search PubMed for article IDs (PMIDs)
    try:
        search_url = f"{base_url}esearch.fcgi?db=pubmed&term={search_term}&retmax={max_results}&sort=relevance"
        response = requests.get(search_url, timeout=15)
        response.raise_for_status()

        root = ET.fromstring(response.content)
        id_list = [elem.text for elem in root.findall('.//Id')]

        if not id_list:
            return "No recent research articles found on PubMed for this topic."

    except requests.exceptions.RequestException as e:
        print(f"Error searching PubMed: {e}")
        return f"Could not connect to the PubMed database to fetch research data. Error: {e}"
    except ET.ParseError as e:
        print(f"Error parsing PubMed search response: {e}")
        return "Failed to parse the response from the PubMed database."

    # Step 2: Fetch the summaries for the found article IDs
    try:
        ids = ",".join(id_list)
        fetch_url = f"{base_url}efetch.fcgi?db=pubmed&id={ids}&retmode=xml&rettype=abstract"
        fetch_response = requests.get(fetch_url, timeout=15)
        fetch_response.raise_for_status()

        fetch_root = ET.fromstring(fetch_response.content)
        articles = fetch_root.findall('.//PubmedArticle')

        summary_texts = []
        for article in articles:
            title_elem = article.find('.//ArticleTitle')
            abstract_elem = article.find('.//AbstractText')
            
            title = title_elem.text if title_elem is not None else "No Title"
            abstract = abstract_elem.text if abstract_elem is not None else "No Abstract"
            
            summary_texts.append(f"  • <b>{title}</b><br>    <i>{abstract[:350]}...</i>")

        return "<br><br>".join(summary_texts) if summary_texts else "Found articles, but could not extract summaries."

    except requests.exceptions.RequestException as e:
        print(f"Error fetching PubMed article details: {e}")
        return "Could not fetch details for the found research articles."
    except ET.ParseError as e:
        print(f"Error parsing PubMed fetch response: {e}")
        return "Failed to parse the detailed article information from PubMed."
