import collections
import glob
import gzip
import logging
import multiprocessing as mp
import os
from functools import partial
from multiprocessing.pool import ThreadPool
import time
from typing import List, Optional, Tuple, Union

import warnings

import pandas as pd
import requests
from alphabet_detector import AlphabetDetector
from cytoolz import pipe, juxt
from gensim.parsing import preprocessing
from fastwarc.warc import ArchiveIterator, WarcRecordType
from rbloom import Bloom

from . import utilities

# This dictionary maps country codes to (English) country names
COUNTRY_CODE_NAME = {
    "ad": "Andorra", "ae": "United_Arab_Emirates", "af": "Afghanistan", "ag": "Antigua_and_Barbuda", "al": "Albania",
    "am": "Armenia", "ao": "Angola", "aq": "Antarctica", "ar": "Argentina", "as": "American_Samoa", "at": "Austria",
    "au": "Australia", "aw": "Aruba", "ax": "Åland", "az": "Azerbaijan", "ba": "Bosnia and Herzegovina",
    "bb": "Barbados", "bd": "Bangladesh", "be": "Belgium", "bf": "Burkina_Faso", "bg": "Bulgaria", "bh": "Bahrain",
    "bi": "Burundi", "bj": "Benin", "bl": "Saint_Barthélemy", "bm": "Bermuda", "bn": "Brunei ", "bo": "Bolivia",
    "bq": "Caribbean_Netherlands", "br": "Brazil", "bs": "Bahamas", "bt": "Bhutan", "bw": "Botswana", "by": "Belarus",
    "bz": "Belize", "ca": "Canada", "cc": "Cocos", "cd": "Democratic_Republic_Congo", "cf": "Central_African_Republic",
    "cg": "Republic_of_Congo", "ch": "Switzerland", "ci": "Côte_d'Ivoire", "ck": "Cook_Islands", "cl": "Chile",
    "cm": "Cameroon", "cn": "China", "co": "Colombia", "cr": "Costa_Rica", "cu": "Cuba", "cv": "Cabo_Verde",
    "cw": "Curaçao", "cx": "Christmas_Island", "cy": "Cyprus", "cz": "Czechia", "de": "Germany", "dj": "Djibouti",
    "dk": "Denmark", "dm": "Dominica", "do": "Dominican_Republic", "dz": "Algeria", "ec": "Ecuador", "ee": "Estonia",
    "eg": "Egypt", "er": "Eritrea", "es": "Spain", "et": "Ethiopia", "fi": "Finland", "fj": "Fiji",
    "fk": "Falkland_Islands", "fm": "Federated_States_Micronesia", "fo": "Faroe_Islands", "fr": "France", "ga": "Gabon",
    "gb": "United_Kingdom ", "gd": "Grenada", "ge": "Georgia", "gf": "French_Guiana", "gg": "Guernsey", "gh": "Ghana",
    "gi": "Gibraltar", "gl": "Greenland", "gm": "Gambia", "gn": "Guinea", "gp": "Guadeloupe", "gq": "Equatorial_Guinea",
    "gr": "Greece", "gs": "South_Georgia", "gt": "Guatemala", "gu": "Guam", "gw": "Guinea-Bissau", "gy": "Guyana",
    "hk": "Hong_Kong", "hm": "Heard_Island", "hn": "Honduras", "hr": "Croatia", "ht": "Haiti", "hu": "Hungary",
    "id": "Indonesia", "ie": "Ireland", "il": "Israel", "im": "Isle_of_Man", "in": "India", "iq": "Iraq", "ir": "Iran",
    "is": "Iceland", "it": "Italy", "je": "Jersey", "jm": "Jamaica", "jo": "Jordan", "jp": "Japan", "ke": "Kenya",
    "kg": "Kyrgyzstan", "kh": "Cambodia", "ki": "Kiribati", "km": "Comoros", "kn": "Saint_Kitts_Nevis",
    "kp": "North_Korea", "kr": "South_Korea", "kw": "Kuwait", "ky": "Cayman_Islands", "kz": "Kazakhstan", "la": "Lao",
    "lb": "Lebanon", "lc": "Saint_Lucia", "li": "Liechtenstein", "lk": "Sri_Lanka", "lr": "Liberia", "ls": "Lesotho",
    "lt": "Lithuania", "lu": "Luxembourg", "lv": "Latvia", "ly": "Libya", "ma": "Morocco", "mc": "Monaco",
    "md": "Moldova", "me": "Montenegro", "mf": "Saint-Martin", "mg": "Madagascar", "mh": "Marshall_Islands ",
    "mk": "North_Macedonia", "ml": "Mali", "mm": "Myanmar", "mn": "Mongolia", "mo": "Macao",
    "mp": "Northern_Mariana_Islands", "mq": "Martinique", "mr": "Mauritania", "ms": "Montserrat", "mt": "Malta",
    "mu": "Mauritius", "mv": "Maldives", "mw": "Malawi", "mx": "Mexico", "my": "Malaysia", "mz": "Mozambique",
    "na": "Namibia", "nc": "New_Caledonia", "ne": "Niger", "nf": "Norfolk_Island", "ng": "Nigeria", "ni": "Nicaragua",
    "nl": "Netherlands", "no": "Norway", "np": "Nepal", "nr": "Nauru", "nu": "Niue", "nz": "New_Zealand", "om": "Oman",
    "pa": "Panama", "pe": "Peru", "pf": "French_Polynesia", "pg": "Papua_New_Guinea", "ph": "Philippines",
    "pk": "Pakistan", "pl": "Poland", "pm": "Saint_Pierre", "pn": "Pitcairn", "pr": "Puerto Rico", "ps": "Palestine",
    "pt": "Portugal", "pw": "Palau", "py": "Paraguay", "qa": "Qatar", "re": "Réunion", "ro": "Romania", "rs": "Serbia",
    "ru": "Russia", "rw": "Rwanda", "sa": "Saudi_Arabia", "sb": "Solomon_Islands", "sc": "Seychelles", "sd": "Sudan",
    "se": "Sweden", "sg": "Singapore", "sh": "Saint_Helena", "si": "Slovenia", "sk": "Slovakia", "sl": "Sierra_Leone",
    "sm": "San_Marino", "sn": "Senegal", "so": "Somalia", "sr": "Suriname", "ss": "South_Sudan", "st": "Sao_Tome",
    "sv": "El_Salvador", "sx": "Sint_Maarten", "sy": "Syria", "sz": "Eswatini", "tc": "Caicos_Islands", "td": "Chad",
    "tf": "French_Southern", "tg": "Togo", "th": "Thailand", "tj": "Tajikistan", "tk": "Tokelau", "tl": "Timor-Leste",
    "tm": "Turkmenistan", "tn": "Tunisia", "to": "Tonga", "tp": "East_Timor", "tr": "Turkey", "tt": "Trinidad_Tobago",
    "tv": "Tuvalu", "tw": "China", "tz": "Tanzania", "ua": "Ukraine", "ug": "Uganda", "uk": "United_Kingdom",
    "us": "United_States", "uy": "Uruguay", "uz": "Uzbekistan", "va": "The_Vatican", "vc": "Saint_Vincent",
    "ve": "Venezuela", "vg": "Virgin_Islands", "vi": "Virgin_Islands", "vn": "Viet_Nam", "vu": "Vanuatu",
    "wf": "Wallis_Futuna", "ws": "Samoa", "ye": "Yemen", "yt": "Mayotte", "za": "South_Africa", "zm": "Zambia",
    "zw": "Zimbabwe", "ελ": "Greece", "бг": "Bulgaria", "бел": "Bulgaria", "мкд": "North_Macedonia", "рф": "Russia",
    "срб": "Serbia", "укр": "Ukraine", "қаз": "Kazakhstan", "հայ": "Armenia", "الاردن": "Jordan", "الجزائر": "Algeria",
    "السعودية": "Saudi_Arabia", "المغرب": "Morocco", "امارات": "United_Arab_Emirates", "ایران": "Iran",
    "بھارت": "India", "تونس": "Tunisia", "سودان": "Sudan", "سورية": "Syria", "عراق": "Iraq", "عمان": "Oman",
    "فلسطين": "Palestine", "قطر": "Qatar", "مصر": "Egypt", "مليسيا": "Malaysia", "موريتانيا": "Mauritania",
    "پاكستان": "Pakistan", "پاکستان": "Pakistan", "ڀارت": "India", "भारत": "India", "বাংলা": "Bangladesh",
    "ভারত": "India", "ਭਾਰਤ": "India", "ભારત": "India", "இந்தியா": "India", "இலங்கை": "Sri_Lanka",
    "சிங்கப்பூர்": "Singapore", "భారత్": "India", "ಭಾರತ": "India", "ഭാരതം": "India", "ලංකා": "Thailand",
    "ไทย": "Thailand", "中国": "China", "中國": "China", "台湾": "Taiwan", "台灣": "Taiwan", "新加坡": "Singapore",
    "澳門": "Macao", "香港": "Hong_Kong", "한국": "South_Korea"
}
# This dictionary maps country names to large regions for organizational purposes
COUNTRY_CODE_REGION = {
    "ad": "europe_west", "ae": "middle_east", "af": "asia_central", "al": "europe_west", "ao": "africa_sub",
    "aq": "antarctica", "ar": "america_south", "as": "asia_southeast", "at": "europe_west", "au": "oceania",
    "aw": "america_central", "ax": "europe_west", "az": "asia_central", "ba": "europe_east", "bb": "america_central",
    "bd": "asia_south", "be": "europe_west", "bf": "africa_sub", "bg": "europe_east", "bh": "middle_east",
    "bi": "africa_sub", "bj": "africa_sub", "bl": "american_central", "bm": "america_central", "bn": "asia_southeast",
    "bo": "america_south", "bq": "america_central", "br": "america_brazil", "bs": "america_central", "bt": "asia_south",
    "bv": "europe_west", "bw": "africa_southern", "by": "europe_east", "bz": "america_central", "ca": "america_north",
    "cd": "africa_sub", "cf": "africa_sub", "cg": "africa_sub", "ch": "europe_west", "ci": "africa_sub",
    "ck": "asia_southeast", "cl": "america_south", "cm": "africa_sub", "cn": "asia_east", "co": "america_south",
    "cr": "america_central", "cu": "america_central", "cv": "africa_sub", "cw": "america_central",
    "cx": "asia_southeast", "cy": "europe_west", "cz": "europe_east", "de": "europe_west", "dj": "africa_north",
    "dk": "europe_west", "dm": "america_central", "do": "america_central", "dz": "africa_north", "ec": "america_south",
    "ee": "europe_east", "eg": "middle_east", "er": "africa_north", "es": "europe_west", "et": "africa_north",
    "fi": "europe_west", "fj": "asia_southeast", "fk": "america_south", "fm": "asia_southeast", "fo": "europe_west",
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
    "pm": "america_north", "pr": "america_central", "ps": "middle_east", "pt": "europe_west", "pw": "asia_southeast",
    "py": "america_south", "qa": "middle_east", "re": "africa_sub", "ro": "europe_east", "rs": "europe_east",
    "ru": "europe_russia", "rw": "africa_sub", "sa": "middle_east", "sb": "asia_southeast", "sc": "asia_south",
    "sd": "africa_north", "se": "europe_west", "sg": "asia_southeast", "si": "europe_east", "sk": "europe_east",
    "sl": "africa_sub", "sm": "asia_southeast", "sn": "africa_sub", "so": "africa_north", "sr": "america_south",
    "ss": "africa_sub", "su": "europe_russia", "sv": "america_central", "sx": "america_central", "sy": "middle_east",
    "sz": "africa_southern", "tc": "america_central", "td": "africa_sub", "tg": "africa_sub", "th": "asia_southeast",
    "tj": "asia_central", "tl": "asia_southeast", "tm": "asia_central", "tn": "africa_north", "tp": "asia_southeast",
    "tr": "middle_east", "tt": "america_central", "tw": "asia_east", "tz": "africa_sub", "ua": "europe_east",
    "ug": "africa_sub", "uk": "europe_west", "us": "america_north", "uy": "america_south", "uz": "asia_central",
    "va": "europe_west", "vc": "america_central", "ve": "america_south", "vg": "america_central",
    "vi": "america_central", "vn": "asia_southeast", "vu": "asia_southeast", "wf": "asia_southeast",
    "ye": "middle_east", "yt": "africa_sub", "za": "africa_southern", "zm": "africa_sub", "zw": "africa_southern",
    "ελ": "europe_west", "бг": "europe_east", "бел": "europe_east", "мкд": "europe_east", "рф": "europe_russia",
    "срб": "europe_east", "укр": "europe_east", "қаз": "asia_central", "հայ": "asia_central", "الاردن": "middle_east",
    "الجزائر": "africa_north", "السعودية": "middle_east", "المغرب": "middle_east", "امارات": "middle_east",
    "ایران": "middle_east", "بھارت": "asia_south", "تونس": "africa_north", "سودان": "africa_sub",
    "سورية": "middle_east", "عراق": "middle_east", "عمان": "middle_east", "فلسطين": "middle_east", "قطر": "middle_east",
    "مصر": "middle_east", "مليسيا": "asia_southeast", "موريتانيا": "africa_north", "پاكستان": "asia_south",
    "پاکستان": "asia_south", "ڀارت": "asia_south", "भारत": "asia_south", "বাংলা": "asia_south", "ভারত": "asia_south",
    "ਭਾਰਤ": "asia_south", "ભારત": "asia_south", "இந்தியா": "asia_south", "இலங்கை": "asia_south",
    "சிங்கப்பூர்": "asia_southeast", "భారత్": "asia_south", "ಭಾರತ": "asia_south", "ഭാരതം": "asia_south",
    "ලංකා": "asia_southeast", "ไทย": "asia_southeast", "中国": "asia_east", "中國": "asia_east", "台湾": "asia_east",
    "台灣": "asia_east", "新加坡": "asia_southeast", "澳門": "asia_east", "香港": "asia_east", "한국": "asia_east",
    "st": "africa_sub"
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
                 url_filter: Optional[Union[str, os.PathLike, bytes]] = None,
                 download_dir: Union[str, os.PathLike, bytes] = "./common_crawl_download"):

        # Ignore certain countries if there is already enough data
        if countries_to_skip is None:
            countries_to_skip = []
        self.countries_to_skip = countries_to_skip

        # Url filter list from file
        self.url_filter = []
        if url_filter is not None:
            self.url_filter = utilities.get_url_filters_from_file(url_filter).keys()

        # Download directory
        self.download_dir = download_dir

        # Detecting language based on alphabet
        self.alphabet_detector = AlphabetDetector()

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

    # ----------------------------------------------------------------------------------------------#
    def _process_wet_record(self, wet_record) -> Optional[List[Tuple[str, str, str, int, str, int]]]:
        """Read individual wet record, split the content to different paragraph, apply filter to remove unwanted
        character and short/trivial lines """
        if wet_record.record_type != WarcRecordType.conversion:
            return
        url = wet_record.headers.get("WARC-Target-URI")
        # getting domain abc.example.com -> ExtractResult(subdomain='abc', domain='hostname', suffix='com')
        url_domain, url_suffix = utilities.extract_url(url)

        if url_suffix not in COUNTRY_CODE_NAME.keys() or url_domain in self.url_filter:
            return
        current_country = COUNTRY_CODE_NAME.get(url_suffix)

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
            # Check if line has Chinese / Japanese / Korean characters, then set length to 15:
            if any(juxt(self.alphabet_detector.is_cjk,
                        self.alphabet_detector.is_hangul,
                        self.alphabet_detector.is_hiragana,
                        self.alphabet_detector.is_katakana
                        )(line)):
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
                processed_line.append((url_suffix, current_country, url, line_num, line))
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

        df = pd.DataFrame(lines, columns=("Domain", "Country", "URL", "LineID", "Text"))
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

    def _deduplicate_cc(self, bf: Bloom, metadata_df: pd.DataFrame, path_to_input: str, path_to_output=None):
        """This method conducts deduplication on a directory of crawl files"""
        self.logger.info(f'_deduplicate_cc: De-duplicating {os.path.basename(path_to_input)}')
        warnings.simplefilter(action="ignore", category=FutureWarning)

        df = pd.read_feather(path_to_input)
        if path_to_output is None:
            path_to_output = path_to_input
        original_len = len(df.index)

        victims = []
        ad = AlphabetDetector()

        for index, document in df.iterrows():
            doc_text = document["Text"]
            scripts = ad.detect_alphabet(doc_text)
            if doc_text in bf:
                victims.append(index)
                for script in scripts:
                    if (metadata_df['Script'] == script).any():
                        metadata_df.loc[metadata_df['Script'] == script]['Deleted'] += 1 # TODO: CHANGE THIS OR ELSE
                    else:
                        metadata_df.loc[metadata_df.shape[0]] = [script, 0, 1]
            else:
                bf.add(doc_text)
                for script in scripts:
                    if (metadata_df['Script'] == script).any():
                        metadata_df.loc[metadata_df['Script'] == script]['Kept'] += 1
                    else:
                        metadata_df.loc[df.shape[0]] = [script, 1, 0]

        df.drop(index=victims)

        self.logger.debug(
            f"_deduplicate_cc: {original_len} formatted and removed {len(victims)}, remaining: {len(df.index)}")
        
        df.to_feather(path_to_output)
        self.logger.debug(f'_deduplicate_cc: saved as {path_to_output}')

        # ------------------------------------------------------------------------------------------------------------#

    def automatically_process_crawl(self, prefix_list, chunk_size=2, max_chunks=-1, dedup_size=1):
        """Automatically download, process, and deduplicate on 1 prefix
        e.g. CC-MAIN-2022-40
        """
        self.logger.debug(f'automatically_process_crawl: begin processing on {prefix_list}')
        prefix_filedir = self.download_cc(prefix_list)
        with gzip.open(prefix_filedir) as index_file:
            lines = [line.decode("utf-8").rstrip() for line in index_file.readlines()]
        chunks = utilities.divide_list(lines, chunk_size)
        bf = Bloom(1000000 * dedup_size, 0.0001)
        metadata_df = pd.DataFrame(columns=["Script", "Kept", "Deleted"])

        # process each chunk
        for i, chunk in enumerate(chunks, start=1):
            self.logger.info(f'Processing chunk {i} of {len(chunks)}')
            self.logger.debug(chunk)
            for segment in chunk:
                self.download_and_process_wet_segment(segment)

            # Combine all dataframe within a shard
            df_files = glob.glob(os.path.join(self.download_dir, prefix_list, 'CC-MAIN-*.feather'))
            df_list = []
            for df_file in df_files:
                df_list.append(pd.read_feather(df_file))
                os.remove(df_file)
            # Save to file using the latest name, add prefix combined
            filename = os.path.join(self.download_dir, prefix_list, f"combined-{os.path.basename(max(df_files))}")
            pd.concat(df_list, ignore_index=True).to_feather(filename)
            
            # Dedupe, add prefix deduplicated
            new_filename = os.path.join(self.download_dir, prefix_list, f"deduplicated-{os.path.basename(filename)}")
            

            self._deduplicate_cc(bf, metadata_df, filename, new_filename)
            os.remove(filename)
            if i % dedup_size == 0:
                bf.clear()
            if i == max_chunks:
                break

        metadata_path = os.path.join(self.download_dir, prefix_list, "metadata.csv")
        metadata_df.to_csv(metadata_path)
        self.logger.debug(f'automatically_process_crawl: saved metadata to {metadata_path}')

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

    def scan_url_filters(self, dataframe_dir: Union[str, os.PathLike, bytes]):
        # todo get url filter from file
        df = pd.read_feather(dataframe_dir)
        # todo scan url from each dataframe
            # todo keep track of the domain and its language, if domain contain more than 3 language, add it onto the file
        # todo pickle the dictionary so that we can keep track of the number of language each domian have, and how many pages it got
        pass

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
