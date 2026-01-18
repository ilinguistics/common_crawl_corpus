import collections
import glob
import gzip
import logging
import pickle
import multiprocessing as mp
import os
from shutil import rmtree
from functools import partial
from multiprocessing.pool import ThreadPool
from typing import List, Optional, Tuple, Union, Dict
from collections import Counter

import warnings

import pandas as pd
import requests
from cytoolz import pipe, juxt
from pandas.errors import SettingWithCopyWarning
from gensim.parsing import preprocessing
from fastwarc.warc import ArchiveIterator, WarcRecordType
from rbloom import Bloom
from GlotScript import sp
from geoLid import geoLid, download_model

from . import utilities
from . import url_filter

# This dictionary maps country codes to (English) country names
COUNTRY_CODE_NAME = {
    "ad": "Andorra", "ae": "United_Arab_Emirates", "af": "Afghanistan", "ag": "Antigua_and_Barbuda", "al": "Albania",
    "am": "Armenia", "ao": "Angola", "ar": "Argentina", "as": "American_Samoa", "at": "Austria",
    "au": "Australia", "aw": "Aruba", "ax": "Åland", "az": "Azerbaijan", "ba": "Bosnia and Herzegovina",
    "bb": "Barbados", "bd": "Bangladesh", "be": "Belgium", "bf": "Burkina_Faso", "bg": "Bulgaria", "bh": "Bahrain",
    "bi": "Burundi", "bj": "Benin", "bm": "Bermuda", "bn": "Brunei ", "bo": "Bolivia",
    "br": "Brazil", "bs": "Bahamas", "bt": "Bhutan", "bw": "Botswana", "by": "Belarus",
    "bz": "Belize", "ca": "Canada", "cc": "Cocos", "cd": "Democratic_Republic_Congo", "cf": "Central_African_Republic",
    "cg": "Republic_of_Congo", "ch": "Switzerland", "ci": "Côte_d'Ivoire", "ck": "Cook_Islands", "cl": "Chile",
    "cm": "Cameroon", "cn": "China", "cr": "Costa_Rica", "cu": "Cuba", "cv": "Cabo_Verde",
    "cw": "Curaçao", "cx": "Christmas_Island", "cy": "Cyprus", "cz": "Czechia", "de": "Germany", "dj": "Djibouti",
    "dk": "Denmark", "dm": "Dominica", "do": "Dominican_Republic", "dz": "Algeria", "ec": "Ecuador", "ee": "Estonia",
    "eg": "Egypt", "er": "Eritrea", "es": "Spain", "et": "Ethiopia", "fi": "Finland", "fj": "Fiji",
    "fk": "Falkland_Islands", "fo": "Faroe_Islands", "fr": "France", "ga": "Gabon",
    "gb": "United_Kingdom ", "gd": "Grenada", "ge": "Georgia", "gf": "French_Guiana", "gh": "Ghana",
    "gi": "Gibraltar", "gl": "Greenland", "gm": "Gambia", "gn": "Guinea", "gp": "Guadeloupe", "gq": "Equatorial_Guinea",
    "gr": "Greece", "gs": "South_Georgia", "gt": "Guatemala", "gu": "Guam", "gw": "Guinea-Bissau", "gy": "Guyana",
    "hk": "Hong_Kong", "hm": "Heard_Island", "hn": "Honduras", "hr": "Croatia", "ht": "Haiti", "hu": "Hungary",
    "id": "Indonesia", "ie": "Ireland", "il": "Israel", "im": "Isle_of_Man", "in": "India", "iq": "Iraq", "ir": "Iran",
    "is": "Iceland", "it": "Italy", "je": "Jersey", "jm": "Jamaica", "jo": "Jordan", "jp": "Japan", "ke": "Kenya",
    "kg": "Kyrgyzstan", "kh": "Cambodia", "ki": "Kiribati", "km": "Comoros", "kn": "Saint_Kitts_Nevis",
    "kp": "North_Korea", "kr": "South_Korea", "kw": "Kuwait", "ky": "Cayman_Islands", "kz": "Kazakhstan", "la": "Laos",
    "lb": "Lebanon", "lc": "Saint_Lucia", "li": "Liechtenstein", "lk": "Sri_Lanka", "lr": "Liberia", "ls": "Lesotho",
    "lt": "Lithuania", "lu": "Luxembourg", "lv": "Latvia", "ly": "Libya", "ma": "Morocco", "mc": "Monaco",
    "md": "Moldova", "mf": "Saint-Martin", "mg": "Madagascar", "mh": "Marshall_Islands ",
    "mk": "North_Macedonia", "ml": "Mali", "mm": "Myanmar", "mn": "Mongolia", "mo": "Macao",
    "mp": "Northern_Mariana_Islands", "mq": "Martinique", "mr": "Mauritania", "ms": "Montserrat", "mt": "Malta",
    "mu": "Mauritius", "mv": "Maldives", "mw": "Malawi", "mx": "Mexico", "my": "Malaysia", "mz": "Mozambique",
    "na": "Namibia", "nc": "New_Caledonia", "ne": "Niger", "nf": "Norfolk_Island", "ng": "Nigeria", "ni": "Nicaragua",
    "nl": "Netherlands", "no": "Norway", "np": "Nepal", "nr": "Nauru", "nu": "Niue", "nz": "New_Zealand", "om": "Oman",
    "pa": "Panama", "pe": "Peru", "pf": "French_Polynesia", "pg": "Papua_New_Guinea", "ph": "Philippines",
    "pk": "Pakistan", "pl": "Poland", "pn": "Pitcairn", "pr": "Puerto Rico", "ps": "Palestine",
    "pt": "Portugal", "pw": "Palau", "py": "Paraguay", "qa": "Qatar", "ro": "Romania", "rs": "Serbia",
    "ru": "Russia", "rw": "Rwanda", "sa": "Saudi_Arabia", "sb": "Solomon_Islands", "sc": "Seychelles", "sd": "Sudan",
    "se": "Sweden", "sg": "Singapore", "si": "Slovenia", "sk": "Slovakia", "sl": "Sierra_Leone",
    "sm": "San_Marino", "sn": "Senegal", "so": "Somalia", "sr": "Suriname", "ss": "South_Sudan",
    "su": "Soviet_Union", "sv": "El_Salvador", "sy": "Syria", "sz": "Eswatini", "td": "Chad",
    "tg": "Togo", "th": "Thailand", "tj": "Tajikistan",
    "tm": "Turkmenistan", "tn": "Tunisia", "tr": "Turkey", "tt": "Trinidad_Tobago",
    "tw": "Taiwan", "tz": "Tanzania", "ua": "Ukraine", "ug": "Uganda", "uk": "United_Kingdom",
    "us": "United_States", "uy": "Uruguay", "uz": "Uzbekistan", "va": "The_Vatican",
    "ve": "Venezuela", "vg": "British_Virgin_Islands", "vi": "US_Virgin_Islands", "vn": "Vietnam", "vu": "Vanuatu",
    "ye": "Yemen", "yt": "Mayotte", "za": "South_Africa", "zm": "Zambia",
    "zw": "Zimbabwe", "ελ": "Greece", "бг": "Bulgaria", "бел": "Belarus", "мкд": "North_Macedonia", "рф": "Russia",
    "срб": "Serbia", "укр": "Ukraine", "қаз": "Kazakhstan", "հայ": "Armenia", "الاردن": "Jordan", "الجزائر": "Algeria",
    "السعودية": "Saudi_Arabia", "المغرب": "Morocco", "امارات": "United_Arab_Emirates", "ایران": "Iran",
    "بھارت": "India", "تونس": "Tunisia", "سودان": "Sudan", "سورية": "Syria", "عراق": "Iraq", "عمان": "Oman",
    "فلسطين": "Palestine", "قطر": "Qatar", "مصر": "Egypt", "سوريا": "Syria", "اليمن": "Yemen", "مليسيا": "Malaysia", "موريتانيا": "Mauritania",
    "پاكستان": "Pakistan", "پاکستان": "Pakistan", "ڀارت": "India", "भारत": "India", "বাংলা": "Bangladesh",
    "ভারত": "India", "ਭਾਰਤ": "India", "ભારત": "India", "இந்தியா": "India", "இலங்கை": "Sri_Lanka",
    "சிங்கப்பூர்": "Singapore", "భారత్": "India", "ಭಾರತ": "India", "ഭാരതം": "India", "ලංකා": "Sri_Lanka",
    "ไทย": "Thailand", "中国": "China", "中國": "China", "台湾": "Taiwan", "台灣": "Taiwan", "新加坡": "Singapore",
    "澳門": "Macao", "香港": "Hong_Kong", "한국": "South_Korea"
}
# This dictionary maps country names to large regions for organizational purposes
COUNTRY_CODE_REGION = {
    "ad": "europe_west", "ae": "middle_east", "af": "asia_central", "al": "europe_west", "ao": "africa_sub",
    "ar": "america_south", "as": "asia_southeast", "at": "europe_west", "au": "oceania",
    "aw": "america_central", "ax": "europe_west", "az": "asia_central", "ba": "europe_east", "bb": "america_central",
    "bd": "asia_south", "be": "europe_west", "bf": "africa_sub", "bg": "europe_east", "bh": "middle_east",
    "bi": "africa_sub", "bj": "africa_sub", "bm": "america_central", "bn": "asia_southeast",
    "bo": "america_south", "br": "america_brazil", "bs": "america_central", "bt": "asia_south",
    "bv": "europe_west", "bw": "africa_southern", "by": "europe_east", "bz": "america_central", "ca": "america_north",
    "cd": "africa_sub", "cf": "africa_sub", "cg": "africa_sub", "ch": "europe_west", "ci": "africa_sub",
    "ck": "asia_southeast", "cl": "america_south", "cm": "africa_sub", "cn": "asia_east",
    "cr": "america_central", "cu": "america_central", "cv": "africa_sub", "cw": "america_central",
    "cx": "asia_southeast", "cy": "europe_west", "cz": "europe_east", "de": "europe_west", "dj": "africa_north",
    "dk": "europe_west", "dm": "america_central", "do": "america_central", "dz": "africa_north", "ec": "america_south",
    "ee": "europe_east", "eg": "middle_east", "er": "africa_north", "es": "europe_west", "et": "africa_north",
    "fi": "europe_west", "fj": "asia_southeast", "fk": "america_south", "fo": "europe_west",
    "fr": "europe_west", "ga": "africa_sub", "gb": "europe_west", "gd": "america_central", "ge": "asia_central",
    "gf": "america_south", "gh": "africa_sub", "gi": "africa_north", "gl": "europe_west", "gm": "africa_sub",
    "gn": "africa_sub", "gp": "america_central", "gr": "europe_west", "gt": "america_central", "gu": "oceania",
    "gw": "africa_sub", "gy": "america_south", "hk": "asia_east", "hn": "america_central", "hr": "europe_east",
    "ht": "america_central", "hu": "europe_east", "id": "asia_southeast", "ie": "europe_west", "il": "middle_east",
    "im": "europe_west", "in": "asia_south", "iq": "middle_east", "ir": "asia_central", "is": "europe_west",
    "it": "europe_west", "je": "europe_west", "jm": "america_central", "jo": "middle_east", "jp": "asia_east",
    "ke": "africa_sub", "kg": "asia_central", "kh": "asia_southeast", "ki": "asia_southeast", "km": "africa_sub",
    "kn": "america_central", "kp": "asia_east", "kr": "asia_east", "kw": "middle_east", "ky": "america_central",
    "kz": "asia_central", "lb": "middle_east", "lc": "america_central", "li": "europe_west", "lk": "asia_south",
    "lr": "africa_sub", "ls": "africa_southern", "lt": "europe_east", "lu": "europe_west", "lv": "europe_east",
    "ma": "africa_north", "mc": "europe_west", "md": "europe_east", "mf": "america_central", "mg": "africa_sub",
    "mh": "oceania", "mk": "europe_east", "ml": "africa_sub", "mm": "asia_southeast", "mn": "asia_east",
    "mo": "asia_east", "mp": "oceania", "mq": "america_central", "mr": "africa_sub", "mt": "europe_west",
    "mu": "asia_southeast", "mv": "europe_west", "mw": "africa_sub", "mx": "america_central", "my": "asia_southeast",
    "mz": "africa_sub", "na": "africa_southern", "nc": "oceania", "ne": "africa_sub", "nf": "oceania",
    "ng": "africa_sub", "ni": "america_central", "nl": "europe_west", "no": "europe_west", "np": "asia_south",
    "nr": "asia_southeast", "nz": "oceania", "om": "middle_east", "pa": "america_central", "pe": "america_south",
    "pf": "asia_southeast", "pg": "asia_southeast", "ph": "asia_southeast", "pk": "asia_south", "pl": "europe_east",
    "pr": "america_central", "ps": "middle_east", "pt": "europe_west", "pw": "asia_southeast",
    "py": "america_south", "qa": "middle_east", "ro": "europe_east", "rs": "europe_east",
    "ru": "europe_russia", "rw": "africa_sub", "sa": "middle_east", "sb": "asia_southeast", "sc": "asia_south",
    "sd": "africa_north", "se": "europe_west", "sg": "asia_southeast", "si": "europe_east", "sk": "europe_east",
    "sl": "africa_sub", "sm": "asia_southeast", "sn": "africa_sub", "so": "africa_north", "sr": "america_south",
    "ss": "africa_sub", "su": "europe_russia", "sv": "america_central", "sy": "middle_east",
    "sz": "africa_southern", "td": "africa_sub", "tg": "africa_sub", "th": "asia_southeast",
    "tj": "asia_central", "tl": "asia_southeast", "tm": "asia_central", "tn": "africa_north",
    "tr": "middle_east", "tt": "america_central", "tw": "asia_east", "tz": "africa_sub", "ua": "europe_east",
    "ug": "africa_sub", "uk": "europe_west", "us": "america_north", "uy": "america_south", "uz": "asia_central",
    "va": "europe_west", "ve": "america_south", "vg": "america_central",
    "vi": "america_central", "vn": "asia_southeast", "vu": "asia_southeast",
    "ye": "middle_east", "yt": "africa_sub", "za": "africa_southern", "zm": "africa_sub", "zw": "africa_southern",
    "ελ": "europe_west", "бг": "europe_east", "бел": "europe_east", "мкд": "europe_east", "рф": "europe_russia",
    "срб": "europe_east", "укр": "europe_east", "қаз": "asia_central", "հայ": "asia_central", "الاردن": "middle_east",
    "الجزائر": "africa_north", "السعودية": "middle_east", "المغرب": "middle_east", "امارات": "middle_east",
    "ایران": "middle_east", "بھارت": "asia_south", "تونس": "africa_north", "سودان": "africa_sub",
    "سورية": "middle_east", "عراق": "middle_east", "عمان": "middle_east", "فلسطين": "middle_east", "قطر": "middle_east",
    "مصر": "middle_east", "سوريا": "middle_east", "اليمن": "middle_east", "مليسيا": "asia_southeast", "موريتانيا": "africa_north", "پاكستان": "asia_south",
    "پاکستان": "asia_south", "ڀارت": "asia_south", "भारत": "asia_south", "বাংলা": "asia_south", "ভারত": "asia_south",
    "ਭਾਰਤ": "asia_south", "ભારત": "asia_south", "இந்தியா": "asia_south", "இலங்கை": "asia_south",
    "சிங்கப்பூர்": "asia_southeast", "భారత్": "asia_south", "ಭಾರತ": "asia_south", "ഭാരതം": "asia_south",
    "ලංකා": "asia_southeast", "ไทย": "asia_southeast", "中国": "asia_east", "中國": "asia_east", "台湾": "asia_east",
    "台灣": "asia_east", "新加坡": "asia_southeast", "澳門": "asia_east", "香港": "asia_east", "한국": "asia_east"
}


# ---------------------

def process_lid(segment, input_dir, output_dir):
    # Check if file has been processed
    check = segment.replace("/", ".").replace(".hdf", ".txt")

    if check not in list(os.listdir(os.path.join(".", "check"))):

        print("Starting " + segment)
        from lid.lidNet.lidNet import lidNet
        lid = lidNet(os.path.join("lid", "lidNet", "Models", "Model.LID.MLP.400kx3_hash.1-3grams.262k.hdf"))

        # Load and prepare
        current_df = pd.read_hdf(segment, key="data")

        # Get meta-data
        meta = current_df.iloc[1,]
        current_country = meta["Country"]
        current_region = meta["Region"]

        # Get time
        section = segment.split(".")[2:]
        current_time = ".".join(section).replace(".hdf", "").replace("CC-MAIN-", "")
        current_time_write = current_time
        current_time = current_time[:7]

        text_list = []  # Initialize

        # Join texts by webpage
        for section in current_df.groupby(by="URL"):
            current_url = str(section[0])
            text = section[1].Text.values
            text = "\n".join(text)
            current_size = len(text.split())

            text_list += [(current_time, current_url, current_size, text)]

        current_df = pd.DataFrame(text_list, columns=["Time", "URL", "N_Words", "Text"])
        current_df.loc[:, "Language"] = lid.predict(list(current_df.loc[:, "Text"].values))

        for section in current_df.groupby(by="Language"):
            current_lang = str(section[0])
            write_name = current_region + "." + current_country + "." + current_lang + "." + current_time_write
            os.makedirs(os.path.join(output_dir, current_region, current_country, current_lang), exist_ok=True)
            write_name = os.path.join(output_dir, current_region, current_country, current_lang, write_name)
            section = section[1]
            section.to_csv(write_name + ".gz", header=True, index=False, index_label=False, compression="gzip")

        # Done with all langs
        with open(os.path.join("check", check), "w") as fo:
            fo.write("Done")

        os.remove(segment)
        print("\tDeleted " + segment)

        return


# --------------------

class CC_Corpus(object):

    def __init__(self,
                 countries_to_skip=None,
                 no_eng = False,
                 download_dir: Union[str, os.PathLike, bytes] = "./common_crawl_download",
                 url_dict = {},
                ):

        # Ignore certain countries if there is already enough data
        if countries_to_skip is None:
            countries_to_skip = []
        self.countries_to_skip = countries_to_skip

        #Ignore English if there is already enough data
        self.no_eng = no_eng

        # Download directory
        self.download_dir = download_dir

        # Load or initialize extended url_filter
        if os.path.exists(os.path.join(self.download_dir, "url_dict.p")) and url_dict != False:
            with open(os.path.join(self.download_dir, "url_dict.p"), "rb") as handle:
                self.url_dict = pickle.load(handle)

        else:
            self.url_dict = {}

        #url_list
        self.url_list = url_filter.url_list

        #Update url_list if loaded new url_dict
        if self.url_dict != {}:
            for domain in self.url_dict:
                if len(self.url_dict[domain]) > 3:
                    if domain not in self.url_list:
                        self.url_list.append(domain)

        # This list defines what countries to include in the corpus
        self.country_codes = []

        # this sets up our module level logging, so we can track execution
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.ch = logging.StreamHandler()
        self.ch.setLevel(logging.DEBUG)
        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.ch.setFormatter(self.formatter)
        self.logger.addHandler(self.ch)
        self.logger.debug('cc_corpus: class initialized')

    # setup input and output dirs for methods

    # ------------------------------------------------------------------------------------------------------------#

    def _download_geolid_models(self, model_location: str = None):
        if model_location is None:
            model_location = os.path.join(self.download_dir, "lid_models")

        model_list = ["baseline", "africa_north", "africa_southern", "africa_sub", "america_brazil", "america_central",
                      "america_north", "america_south", "asia_central", "asia_east", "asia_south",
                      "asia_southeast", "europe_east", "europe_russia", "europe_west", "middle_east", "oceania"]

        for model in model_list:
            model_file = os.path.join(model_location, "geolid."+model+".bin")
            if not os.path.exists(model_file):
                os.makedirs(model_location, exist_ok=True)
                download_model(model, data_dir=model_location)
                self.logger.debug(f'_download_geolid_models: saved {model} model as {model_file}')

    # ----------------------------------------------------------------------------------------------#

    def _label_geolid(self, df):
        self.logger.info(f"_label_geolid: labeling {len(df)} lines")

        self._download_geolid_models()
        lid = geoLid(model_location = os.path.join(self.download_dir, "lid_models"))
        labeled_df = pd.DataFrame()

        for region, group in df.groupby("Region"):
            group = group.copy()
            group["Lang_Region"] = lid.predict(data = group["Text"], region = region)
            labeled_df = pd.concat([labeled_df, group], ignore_index=True)

        labeled_df["Lang_Base"] = lid.predict(data = labeled_df["Text"], region = "baseline")

        #Remove English if necessary
        if self.no_eng == True:
            starting = len(labeled_df)
            labeled_df = labeled_df[labeled_df.loc[:,"Lang_Region"] != "eng"]
            print("No English:", starting, len(labeled_df))

        return labeled_df

    # ----------------------------------------------------------------------------------------------#

    def _process_wet_record(self, wet_record) -> Optional[List[Tuple[str, str, str, int, str, List[str]]]]:
        """Read individual wet record, split the content to different paragraph, apply filter to remove unwanted
        character and short/trivial lines """
        if wet_record.record_type != WarcRecordType.conversion:
            return
        url = wet_record.headers.get("WARC-Target-URI")
        # getting domain abc.example.com -> ExtractResult(subdomain='abc', domain='hostname', suffix='com')
        url_domain, url_suffix = utilities.extract_url(url)

        if url_suffix not in COUNTRY_CODE_NAME.keys() or url_domain in self.url_list:
            return
        current_country = COUNTRY_CODE_NAME.get(url_suffix)
        current_region = COUNTRY_CODE_REGION.get(url_suffix)

        web_content: str = wet_record.reader.read().decode("utf-8")
        processed_line: List[Tuple[str, str, str, int, str, int]] = []
        line_num = 0  # flag to make sure it is the same page

        for line in web_content.splitlines():
            # we need the line larger than 15 character
            if len(line) <= 15:
                continue
            line = pipe(line,
                        # Remove links, hashtags, at-mentions, mark-up, and "RT"
                        utilities.strip_tags,
                        # Remove emojis
                        utilities.remove_emoji,
                        # Remove extra spaces
                        preprocessing.strip_tags,
                        preprocessing.split_alphanum,
                        preprocessing.strip_multiple_whitespaces)

            # Check if still above 15 and not contains any navigational / boilerplate characters
            if len(line) <= 15 or any(char in line for char in utilities.ILLEGAL_CHAR):
                continue
            # Check if mostly numbers / characters
            character_only = pipe(line, preprocessing.strip_numeric, preprocessing.strip_punctuation)
            if len(character_only) <= 12:
                continue
            
            script_details = sp(line)[2]["details"]

            if not script_details:
                continue

            scripts = list(script_details.keys())

            # Check if line has Chinese / Japanese / Korean characters, then set length to 15:
            if any(script in ["Hani", "Hans", "Hant", "Hrkt", "Kana", "Hira", "Jpan", "Hang", "Jamo", "Kore"] for script in scripts):
                length = 15
            else:
                length = 50
            if len(line) < length:
                continue
            string_counter = collections.Counter(line)
            if all([string_counter.get("-", 0) < 4, string_counter.get("(", 0) < 4, string_counter.get(")", 0) < 4,
                    string_counter.get("=", 0) < 2, string_counter.get("_", 0) < 2, string_counter.get(".", 0) < 15,
                    string_counter.get("&", 0) < 4, string_counter.get("[", 0) < 3, string_counter.get("]", 0) < 3,
                    string_counter.get("*", 0) < 5]):
                line_num += 1
                processed_line.append((url_suffix, current_country, current_region, url, line_num, line, scripts))

        return processed_line

    def download_and_process_wet_segment(self, index: str):
        """
        Downloads (as stream) the second level index file for the given year range. Doesn't need to save the actual files
        Then, processes returns a dataframe containing the common fields
        e.g. crawl-data/CC-MAIN-2022-40/segments/1664030331677.90/wet/CC-MAIN-20220924151538-20220924181538-00000.warc.wet.gz
        """
        self.logger.debug(f"download & process_wet_segment: processing {os.path.basename(index)}")
        url = f"https://data.commoncrawl.org/{index}".strip()
        segment_stream = requests.get(url, stream=True)

        lines = []
        for record in ArchiveIterator(segment_stream.raw):
            if temp := self._process_wet_record(record):
                lines.extend(temp)
        # add prefix dataframe to filename, change extension to .feather stead of gzip
        path_split = index.split("/")
        cc_index = path_split[1]  # CC-MAIN-2022-40
        name, _ = os.path.splitext(path_split[-1])  # e.g. CC-MAIN-2....wet

        df = pd.DataFrame(lines, columns=("Domain", "Country", "Region", "URL", "LineID", "Text", "Scripts"))
        df.reset_index()
        df.to_feather(os.path.join(self.download_dir, cc_index, f'{name}.feather'))

    # ------------------------------------------------------------------------------------------------#

    def download_cc(self, prefix_list: str) -> str:
        """This method downloads the complete CC for a given prefix, from the path file to the WARC files.
        e.g. CC-MAIN-2022-40
        """
        url = f"https://data.commoncrawl.org/crawl-data/{prefix_list}/wet.paths.gz"
        os.makedirs(os.path.join(self.download_dir, prefix_list), exist_ok=True)
        filepath = os.path.join(self.download_dir, prefix_list, f"{prefix_list}-wet.paths.gz".strip())

        self.logger.info(f'Download_cc: Prefix {prefix_list} downloading, \tsave dir: {filepath}')

        response = requests.get(url)
        with open(filepath, "wb") as file:
            file.write(response.content)
        return filepath

    # ----------------------------------------------------------------------------------------------------------------------#

    def _reduce_metadata(self, df, col_name = "Script"):
        """Aggregate meta-data across many chunks into one file"""

        holder = []

        for name, name_df in df.groupby(col_name):

            kept = name_df.loc[:,"Kept"].sum()
            removed = name_df.loc[:,"Deleted"].sum()
            percent = kept / (kept + removed)
            holder.append([name, kept, removed, percent])

        df = pd.DataFrame(holder, columns = [col_name, "Kept", "Removed", "Pct_Kept"])
        
        return df

    # ----------------------------------------------------------------------------------------------------------------------#

    def _deduplicate_cc(self, bf: Bloom, metadata: Dict[str, Dict[str, Counter]], path_to_input: str, path_to_output: Optional[str] = None):
        """This method conducts deduplication on a feathered DataFrame and saves the result to a new file."""
        warnings.simplefilter(action="ignore", category=FutureWarning)
        warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

        df = pd.read_feather(path_to_input)
        if path_to_output is None:
            path_to_output = path_to_input

        victims = []
        if "script" not in metadata:
            metadata["script"] = {}
        if "lang_region" not in metadata:
            metadata["lang_region"] = {}

        for index, document in df.iterrows():
            doc_text = document["Text"]
            scripts = document["Scripts"]
            lang_region = document["Lang_Region"]
            if doc_text in bf:
                victims.append(index)
                for script in scripts:
                    if script not in metadata["script"]:
                        metadata["script"][script] = Counter()
                    metadata["script"][script]["Deleted"] += 1
                if lang_region not in metadata["lang_region"]:
                    metadata["lang_region"][lang_region] = Counter()
                metadata["lang_region"][lang_region]["Deleted"] += 1
            else:
                bf.add(doc_text)
                for script in scripts:
                    if script not in metadata["script"]:
                        metadata["script"][script] = Counter()
                    metadata["script"][script]["Kept"] += 1
                if lang_region not in metadata["lang_region"]:
                    metadata["lang_region"][lang_region] = Counter()
                metadata["lang_region"][lang_region]["Kept"] += 1

        df.drop(index=victims)        
        df.to_feather(path_to_output)

        # ------------------------------------------------------------------------------------------------------------#
    def _dedupe_one_language(self, df):

        #Initialize
        bf = Bloom(len(df), 0.01)
        holder = []

        #Add each text to filter and check
        for text in df.loc[:,"Text"].values:
            
            if text in bf:
                holder.append("Drop")
            else:
                holder.append("Keep")

            bf.update(text)

        #Add column and filter df
        df.loc[:,"Status"] = holder
        df = df[df.loc[:,"Status"] == "Keep"]
        df = df.drop("Status", axis = 1)

        return df

        # ------------------------------------------------------------------------------------------------------------#

    def automatically_aggregate_crawl(self, prefix):
        """Takes a chunk of the crawl (ie., CC-MAIN-2022-40) and aggregates the current output of automatically_process_crawl,
        applying further deduplication and updating the url filter, then saves to language-specific directories."""

        print(prefix)
        holder = []
        lang_holder = []
        script_holder = []
        to_delete = []
        max_chunk = 0

        #First, set the directory
        root = os.path.join(self.download_dir, prefix)

        #Second, iterate over files
        for file in sorted(os.listdir(root)):
            if file.endswith(".feather") and ".warc" not in file:

                print(prefix, file)

                #Some files fail to load
                try:
                    df = pd.read_feather(os.path.join(root, file))
                    load_ok = True

                except Exception as e:
                    print(e)
                    load_ok = False

                #Only continue if file loaded
                if load_ok == True:

                    #Get base urls
                    if "URL_Domain" not in df.columns:
                        df = self.scan_url_filters(df)
                    holder.append(df)

                    #Get chunk id
                    chunk_id = int(file.split(".")[1])

                    #Keep track of how far into the chunk we have progressed
                    if chunk_id > max_chunk:
                        max_chunk = chunk_id

                    #Load language and script meta-data
                    lang_file = os.path.join(root, "lang_metadata."+prefix+"."+str(chunk_id)+".csv")
                    script_file = os.path.join(root, "script_metadata."+prefix+"."+str(chunk_id)+".csv")

                    lang_df = pd.read_csv(lang_file)
                    lang_holder.append(lang_df)

                    script_df = pd.read_csv(script_file)
                    script_holder.append(script_df)

                    #List of files to delete after processing
                    to_delete += [os.path.join(root, file), lang_file, script_file]

        #Check for enough objects
        if len(holder) > 1:

            #Make into one df
            df = pd.concat(holder)
            del holder

            #Merge and reduce metadata
            lang_df = pd.concat(lang_holder)
            lang_df = self._reduce_metadata(lang_df, col_name = "Language")
            script_df = pd.concat(script_holder)
            script_df = self._reduce_metadata(script_df, col_name = "Script")

            #Save aggregated meta-data
            lang_df.to_csv(os.path.join(root, "language_aggregated."+str(max_chunk)+".csv"))
            script_df.to_csv(os.path.join(root, "script_aggregated."+str(max_chunk)+".csv"))

            print("Combined size", len(df))

            #Update url_list
            for domain in self.url_dict:
                if len(self.url_dict[domain]) > 3:
                    if domain not in self.url_list:
                        self.url_list.append(domain)

            #Apply url filter to data
            df = df[~df.loc[:,"URL_Domain"].isin(self.url_list)]
            print("After URL filter", len(df))

            #Save url_dict
            with open(os.path.join(self.download_dir, "url_dict.p"), "wb") as handle:
                pickle.dump(self.url_dict, handle)

            #Save max chunk reached for further proccessing
            with open(os.path.join(root, "state.txt"), "w") as f:
                f.write(str(max_chunk))

            #Sort into languages
            for language, language_df in df.groupby("Lang_Region"):
                starting = len(language_df)

                #Deduplicate    
                language_df = self._dedupe_one_language(language_df)
                ending = len(language_df)

                #Save to language folder
                os.makedirs(os.path.join(self.download_dir, "!By_Language", language), exist_ok = True)
                language_df.to_feather(os.path.join(self.download_dir, "!By_Language", language, prefix+"."+language+"."+str(max_chunk)+".feather"))
                print(language, starting-ending, end = "\t")

            #Now remove temp chunk files
            print("")
            del df
            print("Deleting files")
            for file in to_delete:
                os.remove(file)

        # ------------------------------------------------------------------------------------------------------------#

    def automatically_process_crawl(self, prefix_list, chunk_size=2, max_chunks=-1, dedup_size=1):
        """Automatically download, process, and deduplicate on 1 prefix
        e.g. CC-MAIN-2022-40
        """
        print("Starting", prefix_list)
        self.logger.debug(f'automatically_process_crawl: begin processing on {prefix_list}')
        prefix_filedir = self.download_cc(prefix_list)
        with gzip.open(prefix_filedir) as index_file:
            lines = [line.decode("utf-8").rstrip() for line in index_file.readlines()]
        chunks = utilities.divide_list(lines, chunk_size)
        bf_map = {}

        #Check if resuming after previous run of automatically_aggregate_crawl
        if os.path.exists(os.path.join(self.download_dir, prefix_list, "state.txt")):
            with open(os.path.join(self.download_dir, prefix_list, "state.txt")) as f:
                current_place = int(f.readlines()[0].strip())
                print("Current chunk starting point:", current_place)

        # process each chunk
        for i, chunk in enumerate(chunks, start=1):

            #First, check if this is finished
            lang_metadata_path = os.path.join(self.download_dir, prefix_list, "lang_metadata."+prefix_list+"."+str(i)+".csv")

            if not os.path.exists(lang_metadata_path) and i > current_place:
                
                metadata = {}

                self.logger.info(f'Processing chunk {i} of {len(chunks)}')
                self.logger.debug(chunk)
                for segment in chunk:

                    try:
                        self.download_and_process_wet_segment(segment)

                    except Exception as e:
                        print(e)

                # Combine all dataframe within a shard
                df_files = glob.glob(os.path.join(self.download_dir, prefix_list, 'CC-MAIN-*.feather'))
                df_list = []
                for df_file in df_files:
                    df_list.append(pd.read_feather(df_file))
                    os.remove(df_file)

                #Check to make sure there are dfs to combine
                if len(df_list) > 1:
                    combined_df = pd.concat(df_list, ignore_index=True)
                    del df_list

                    # GeoLID
                    combined_df = combined_df[combined_df.loc[:,"Region"] != "antarctica"]  #No Antarctica!

                    #Get base urls
                    combined_df = self.scan_url_filters(combined_df)

                    #Apply url filter to data
                    temp_begin = len(combined_df)
                    combined_df = combined_df[~combined_df.loc[:,"URL_Domain"].isin(self.url_list)]

                    if len(combined_df) > 10:
                        geolid_df = self._label_geolid(combined_df)
                        del combined_df
                        
                        holder_dir = os.path.join(self.download_dir, prefix_list+"_temp_lang_dir"+str(i))
                        if not os.path.exists(holder_dir):
                            os.makedirs(holder_dir)

                        # Split by lang
                        for lang in geolid_df["Lang_Region"].unique():
                            lang_df = geolid_df[geolid_df["Lang_Region"] == lang]
                            lang_df.to_feather(os.path.join(holder_dir, f"in.{lang}.feather"))

                        del geolid_df

                        # Deduplicate each language
                        lang_files = [f for f in os.listdir(holder_dir) if f.startswith("in.")]
                        for lang_file in lang_files:
                            lang_file_path = os.path.join(holder_dir, lang_file)
                            new_filename = os.path.join(holder_dir, f"deduplicated-{lang_file}.feather")
                            lang = lang_file.split(".")[1]
                            if lang not in bf_map:
                                bf_map[lang] = Bloom(10000000 * dedup_size, 0.01)
                            bf = bf_map[lang]
                            self._deduplicate_cc(bf, metadata, lang_file_path, new_filename)
                            os.remove(lang_file_path)

                        # Combine all deduplicated files
                        full_df = pd.DataFrame()
                        dedup_files = glob.glob(os.path.join(holder_dir, "deduplicated-*.feather"))
                        df_list = []
                        for dedup_file in dedup_files:
                            df = pd.read_feather(dedup_file)
                            df_list.append(df)
                        full_df = pd.concat(df_list, ignore_index=True)
                        del df_list

                        # Remove everything in holder_dir
                        rmtree(holder_dir)

                        # Save the full deduplicated file
                        new_filename =  os.path.join(self.download_dir, prefix_list, "deduplicated-combined-"+prefix_list+"."+str(i)+".feather")
                        full_df.to_feather(new_filename)
                        self.logger.debug(f'automatically_process_crawl: saved deduplicated file as {new_filename}')

                        # Once the number of chunks reaches the dedup_size, clear the bloom filter to reset deduplication
                        if i % dedup_size == 0:
                            bf_map.clear()
                        if i == max_chunks:
                            break

                        #Save the meta-data for each chunk, to aid easy restarting
                        script_metadata_path = os.path.join(self.download_dir, prefix_list, "script_metadata."+prefix_list+"."+str(i)+".csv")
                        
                        smdf = pd.DataFrame.from_dict(metadata["script"], orient="index").reset_index()
                        smdf.rename(columns={"index": "Script"}, inplace=True)
                        smdf["Percent Removed"] = smdf["Deleted"] / (smdf["Kept"] + smdf["Deleted"])
                        smdf.to_csv(script_metadata_path, index=False)

                        lang_metadata_path = os.path.join(self.download_dir, prefix_list, "lang_metadata."+prefix_list+"."+str(i)+".csv")
                        lmdf = pd.DataFrame.from_dict(metadata["lang_region"], orient="index").reset_index()
                        lmdf.rename(columns={"index": "Language"}, inplace=True)
                        lmdf["Percent Removed"] = lmdf["Deleted"] / (lmdf["Kept"] + lmdf["Deleted"])
                        lmdf.to_csv(lang_metadata_path, index=False)

                        self.logger.debug(f'automatically_process_crawl: saved metadata to {script_metadata_path}')

        # ----------------------------------------------------------------------------------------------------------------------#

    def lid_cc(self, input_dir, output_dir, region, workers):
        """Compare classification of 2 language id models (LID), if it is not the same then remove it"""
        segment_list = []
        for root, dirs, files in os.walk(os.path.join(input_dir, region)):
            for file in files:
                file = os.path.join(root, file)
                segment_list.append(file)

        # Multi-process by file
        pool_instance = mp.Pool(processes=workers, maxtasksperchild=1)
        line_list = pool_instance.map(partial(process_lid,
                                              input_dir=input_dir,
                                              output_dir=output_dir
                                              ), segment_list, chunksize=1)

        pool_instance.close()
        pool_instance.join()

        # ----------------------------------------------------------------------------------------------------------------------#

    def scan_url_filters(self, df):
        
        # Starting list is from the package
        url_list = url_filter.url_list

        # Get base domain and country for each sample
        url_column = []
        for row in df.itertuples():
            url = row[4]
            country = row[2]
            url_domain, url_suffix = utilities.extract_url(url)
            
            #Make sure we are tracking this domain
            if url_domain not in self.url_dict:
                self.url_dict[url_domain] = []

            #Add current country if necessary
            if country not in self.url_dict[url_domain]:
                self.url_dict[url_domain].append(country)

            #Save to column holder
            url_column.append(url_domain)

        #Now add column and return new df
        df.loc[:,"URL_Domain"] = url_column
            
        return df

        # ----------------------------------------------------------------------------------------------------------------------#
    def aggregate_metadata_helper(self, df):

        holder = []

        #For each lang/script
        if "Script" in df.columns:
            name = "Script"
        else:
            name = "Language"

        for name_var, name_df in df.groupby(name):
            for month, month_df in name_df.groupby("Month"):

                kept = month_df.loc[:,"Kept"].sum()
                removed = month_df.loc[:,"Removed"].sum()
                pct = removed / (kept + removed)
                holder.append([name_var, month, kept, removed, pct])

        #Make into df
        df = pd.DataFrame(holder, columns = [name, "Month", "Kept", "Removed", "Pct"])

        return df
        # ----------------------------------------------------------------------------------------------------------------------#

    def aggregate_metadata(self, input_dir):

        lang_list = []      #For language meta-data
        script_list = []    #For script meta-data

        #Iterate over folders
        for folder in os.listdir(input_dir):
            if "." not in folder and "CC-MAIN" in folder:

                print("Starting", folder)
                root = os.path.join(input_dir, folder)

                #Iterate over files
                for file in os.listdir(root):
                    if file.endswith(".csv"):

                        #Load and add month info
                        df = pd.read_csv(os.path.join(root, file), index_col = 0)
                        df.loc[:,"Month"] = folder
                        print(df)

                        #Save
                        if "script" in file:
                            script_list.append(df)
                        elif "language" in file:
                            lang_list.append(df)
        
        #Now concat
        script_df = pd.concat(script_list)
        lang_df = pd.concat(lang_list)

        #Now aggregate
        script_df = self.aggregate_metadata_helper(script_df)
        lang_df = self.aggregate_metadata_helper(lang_df)

        print(script_df)
        print(lang_df)
        
        #Save
        script_df.to_csv("metadata.script.csv")
        lang_df.to_csv("metadata.language.csv")

        return        

        # ----------------------------------------------------------------------------------------------------------------------#

    def final_cc(self, input_dir, output_dir, region):

            for country in os.listdir(os.path.join(input_dir, region)):
                for language in os.listdir(os.path.join(input_dir, region, country)):

                    first_flag = True  # First for this set
                    counter = 1

                    for file in os.listdir(os.path.join(input_dir, region, country, language)):

                        file = os.path.join(input_dir, region, country, language, file)
                        new_df = pd.read_csv(file, compression="gzip")

                        if first_flag == True:
                            first_flag = False
                            current_df = new_df
                            print("\tFirst time for " + region + " " + country + " " + language)

                        else:

                            # First, merge new_df
                            print("\tContinuing with " + file)
                            current_df = pd.concat([current_df, new_df])
                            current_df.drop_duplicates(subset="URL", keep="first", inplace=False)

                            # Second, check length
                            if len(current_df) > 100000:
                                write_df = current_df.head(n=100000)
                                current_df = current_df.tail(n=len(current_df) - 100000)

                                write_name = region + "." + country + "." + language + "." + str(counter) + ".gz"
                                write_name = os.path.join(output_dir, region, country, language, write_name)
                                os.makedirs(os.path.join(output_dir, region, country, language), exist_ok=True)
                                counter += 1

                                write_df.to_csv(write_name, header=True, index=False, index_label=False,
                                                compression="gzip")
                                print("\t\tWriting " + write_name)
                                del write_df

                    # Done with all files, write the remaining
                    write_name = region + "." + country + "." + language + "." + str(counter) + ".gz"
                    write_name = os.path.join(output_dir, region, country, language, write_name)
                    os.makedirs(os.path.join(output_dir, region, country, language), exist_ok=True)
                    counter += 1

                    current_df.to_csv(write_name, header=True, index=False, index_label=False, compression="gzip")
                    print("\t\tWriting " + write_name)

                    del current_df

                    # Done, now delete
                    for file in os.listdir(os.path.join(input_dir, region, country, language)):
                        file = os.path.join(input_dir, region, country, language, file)
                        os.remove(file)

                        # --------------------------------------------------------------------
