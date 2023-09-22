import requests
import json
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import time
import os
import re
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from datetime import datetime
from pathlib import Path

# Marking start time for timing the script.
start_time = time.time()

# Global variables
house_details = []
SCRAPED_URLS = set()
raw_data = []
ERROR_COUNT = 0
URL_COUNT = 0
COUNTER = 0
COUNTER_LOCK = Lock()

# Define data directories using pathlib
DATA_DIR = Path('/opt/airflow/dags/Data')


# Acts as filter for the dictionary, we can add or remove (un)wanted data from the filtered dictionary
selected_values = [
    ("id", "id"),
    ("street", "property.location.street"),
    ("housenumber", "property.location.number"),
    ("box", "property.location.box"),
    ("floor", "property.location.floor"),
    ("city", "property.location.locality"),
    ("postalcode", "property.location.postalCode"),
    ("type", "property.type"),
    ("subtype", "property.subtype"),
    ("location_area", "property.location.type"),
    ("region", "property.location.regionCode"),
    ("district", "property.location.district"),
    ("province", "property.location.province"),    
    ("price", "price.mainValue"),
    ("type_of_sale", "price.type"),
    ("construction_year", "property.building.constructionYear"),
    ("total_surface", "property.land.surface"),
    ("habitable_surface", "property.netHabitableSurface"),
    ("bedroom_count", "property.bedroomCount"),
    ("kitchen_type", "property.kitchen.type"),
    ("furnished", "transaction.sale.isFurnished"),
    ("fireplace", "property.fireplaceExists"),
    ("terrace", "property.hasTerrace"),
    ("terrace_surface", "property.terraceSurface"),
    ("garden", "property.hasGarden"),
    ("garden_surface", "property.gardenSurface"),
    ("facades", "property.building.facadeCount"),
    ("swimmingpool", "property.hasSwimmingPool"),
    ("condition", "property.building.condition"),
    ("epc_score", "transaction.certificates.epcScore"),
    ("latitude", "property.location.latitude"),
    ("longitude", "property.location.longitude"),
    ("property_url", "url")
]
#Start a Sesson as session
session = requests.Session()

def get_property(url, session):
    """
    Fetches the property details from the given URL.

    Args:
        url (str): The URL of the property.
        session (requests.Session): The session object to use for making the GET request.

    Returns:
        dict: The property details as a dictionary.
    """
    try:
        response = session.get(url)
        html_content = response.text
        start_marker = "window.classified = " #create variable for cutting the content string at the start of the dictionary
        end_marker = ";\n"                    #create variable for cutting off the end of the string
        start_index = html_content.find(start_marker) + len(start_marker) 
        end_index = html_content.find(end_marker, start_index)
        if start_index != -1 and end_index != -1: #check if we are not out of bounds with the string
            json_data = html_content[start_index:end_index] #create the dictionary from the resulting string {}
            house_dict = json.loads(json_data) #seperate dict for the filtered result
            house_dict["url"] = url  # Add the URL to the dictionary
            return house_dict, json_data
    except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
        print(f"Error occurred during scraping: {e}")
    return None, None

# Define a function to get property details from a URL
def get_property(url, session):
    try:
        response = session.get(url)
        html_content = response.text
        start_marker = "window.classified = "  # create a variable for cutting the content string at the start of the dictionary
        end_marker = ";\n"  # create a variable for cutting off the end of the string
        start_index = html_content.find(start_marker) + len(start_marker)
        end_index = html_content.find(end_marker, start_index)
        if start_index != -1 and end_index != -1:  # check if we are not out of bounds with the string
            json_data = html_content[start_index:end_index]  # create the dictionary from the resulting string {}
            house_dict = json.loads(json_data)  # separate dict for the filtered result
            house_dict["url"] = url  # Add the URL to the dictionary
            return house_dict, json_data
    except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
        print(f"Error occurred during scraping: {e}")
    return None, None

# Define a function to get property URLs to scrape
def get_urls(num_pages, session):
    list_all_urls = []
    global URL_COUNT  # Add the global variable for URL count
    for i in range(1, num_pages + 1):
        root_url = f"https://www.immoweb.be/en/search/house/for-sale?countries=BE&page={i}&orderBy=relevance"
        req = session.get(root_url)
        content = req.content
        soup1 = BeautifulSoup(content, "html.parser")
        if req.status_code == 200:
            # Find all property URLs on the page and add them to the list
            list_all_urls.extend(tag.get("href") for tag in soup1.find_all("a", attrs={"class": "card__title-link"}))
            print(f'Urls found: {len(list_all_urls)}', end='\r', flush=True)
            URL_COUNT = len(list_all_urls)
        else:
            print("Page not found")
            break
        print(f"Number of properties: {len(list_all_urls)}")

    for i in range(1, num_pages + 1):
        root_url = f"https://www.immoweb.be/en/search/apartment/for-sale?countries=BE&page={i}&orderBy=relevance"
        req = session.get(root_url)
        content = req.content
        soup2 = BeautifulSoup(content, "html.parser")
        if req.status_code == 200:
            # Find all property URLs on the page and add them to the list
            list_all_urls.extend(tag.get("href") for tag in soup2.find_all("a", attrs={"class": "card__title-link"}))
            print(f'Urls found: {len(list_all_urls)}', end='\r', flush=True)
            URL_COUNT = len(list_all_urls)
        else:
            print("Page not found")
            break
        print(f"Number of properties: {len(list_all_urls)}")

    return list_all_urls

# Define a function to save the scraped data
def save_data():
    current_datetime = datetime.now()
    datestamp = current_datetime.strftime("%Y%m%d%H%M%S")
    filename = DATA_DIR / f'scraped_data_{datestamp}.csv'  # Use pathlib

    house_details_df = pd.DataFrame(house_details)

    if not house_details_df.empty:
        house_details_df.replace({np.nan: 0, None: 0}, inplace=True)
        DATA_DIR.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
        house_details_df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")
    else:
        print("No data to save.")

# Define a function to get the latest version
def get_latest_version(file_prefix):
    latest_datestamp = None
    for filename in DATA_DIR.glob(f"{file_prefix}*.csv"):  # Use pathlib
        match = re.search(rf"{file_prefix}(\d+)\.csv", filename.name)
        if match:
            file_datestamp = match.group(1)
            if latest_datestamp is None or file_datestamp > latest_datestamp:
                latest_datestamp = file_datestamp

    return latest_datestamp

def load_data():
    latest_version = get_latest_version("scraped_data_")
    if latest_version is None:
        print("No existing data found.")
        return None  # Return None to indicate that no existing data was found
    else:
        print(f"Existing data found: version {latest_version}.")
        return latest_version  # Return the latest version found




def process_url(url, session):
    """
    Processes a property URL and extracts the relevant details.

    Args:
        url (str): The URL of the property.
    """
    global ERROR_COUNT, URL_COUNT
    if any(record.get("id") == url for record in house_details):
        # Skip if URL already processed
        print(f"Skipping URL: {url}")
        return
    house_dict, raw_json_data = get_property(url, session)
    global SCRAPED_URLS
    if url in SCRAPED_URLS:
        # Skip if URL already processed
        print(f"Skipping URL: {url}")
        return
    if house_dict:
        filtered_house_dict = {}
        for new_key, old_key in selected_values:
            nested_keys = old_key.split(".")
            value = house_dict
            for nested_key in nested_keys:
                if isinstance(value, dict) and nested_key in value:
                    value = value[nested_key]
                else:
                    value = None
                    break
            filtered_house_dict[new_key] = value
        id_match = re.search(r"/(\d+)$", url)
        if id_match:
            filtered_house_dict["id"] = int(id_match.group(1))
        filtered_house_dict["Property url"] = url  # Add "Property url" field
        house_details.append(filtered_house_dict)
        raw_data.append({"url": url, "json_data": raw_json_data})
    else:
        # Increment error count
        ERROR_COUNT += 1
        # Sleep for 3 seconds if property details couldn't be fetched
        time.sleep(3)
        
def process_url_wrapper(url):
    """
    Wrapper function for processing a property URL.

    Args:
        url (str): The URL of the property.
    """
    global COUNTER, URL_COUNT, ERROR_COUNT, SCRAPED_URLS, COUNTER_LOCK
    with COUNTER_LOCK:
        COUNTER += 1
        print(f"URLs processed: {COUNTER}", end='\r', flush=True)
        # Use end='\r' and flush=True to stay on the same line

    process_url(url, session)

def run_scraper(num_pages, num_workers):
    list_of_urls = get_urls(num_pages, session)
    max_threads = min(num_workers, len(list_of_urls))
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        for _ in executor.map(process_url_wrapper, list_of_urls):
            pass
    # Calculate and print the elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Script finished in {elapsed_time:.2f} seconds.")
    print(f"\nTotal URLs processed: {COUNTER}, Total URLs found: {URL_COUNT}, Total errors: {ERROR_COUNT}\n")
    #save the data
    save_data()
    print(f"\nTotal records: {len(house_details)}\n")

if __name__ == "__main__":
    run_scraper(2,10)