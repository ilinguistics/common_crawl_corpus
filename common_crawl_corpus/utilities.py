from typing import Dict
import csv
import emoji
import tldextract
import nltk
import re
import pandas as pd

ILLEGAL_CHAR = ("|", "©", "«", "®", "»", "˂", "˃", "˄", "˅", "/", "\\", "{", "}")

COUNTRY_CODES_NAME: Dict[str, str] = {
    "ad": "Andorra", "ae": "United_Arab_Emirates", "af": "Afghanistan", "ag": "Antigua_and_Barbuda", "al": "Albania",
    "am": "Armenia", "ao": "Angola", "aq": "Antarctica", "ar": "Argentina", "as": "American_Samoa",
    "at": "Austria", "au": "Australia", "aw": "Aruba", "ax": "Åland", "az": "Azerbaijan",
    "ba": "Bosnia and Herzegovina", "bb": "Barbados", "bd": "Bangladesh", "be": "Belgium", "bf": "Burkina_Faso",
    "bg": "Bulgaria", "bh": "Bahrain", "bi": "Burundi", "bj": "Benin", "bl": "Saint_Barthélemy",
    "bm": "Bermuda", "bn": "Brunei ", "bo": "Bolivia", "bq": "Caribbean_Netherlands", "br": "Brazil",
    "bs": "Bahamas", "bt": "Bhutan", "bw": "Botswana", "by": "Belarus", "bz": "Belize",
    "ca": "Canada", "cc": "Cocos", "cd": "Democratic_Republic_Congo", "cf": "Central_African_Republic",
    "cg": "Republic_of_Congo",
    "ch": "Switzerland", "ci": "Côte_d'Ivoire", "ck": "Cook_Islands", "cl": "Chile", "cm": "Cameroon",
    "cn": "China", "co": "Colombia", "cr": "Costa_Rica", "cu": "Cuba", "cv": "Cabo_Verde",
    "cw": "Curaçao", "cx": "Christmas_Island", "cy": "Cyprus", "cz": "Czechia", "de": "Germany",
    "dj": "Djibouti", "dk": "Denmark", "dm": "Dominica", "do": "Dominican_Republic", "dz": "Algeria",
    "ec": "Ecuador", "ee": "Estonia", "eg": "Egypt", "er": "Eritrea", "es": "Spain",
    "et": "Ethiopia", "fi": "Finland", "fj": "Fiji", "fk": "Falkland_Islands", "fm": "Federated_States_Micronesia",
    "fo": "Faroe_Islands", "fr": "France", "ga": "Gabon", "gb": "United_Kingdom ", "gd": "Grenada",
    "ge": "Georgia", "gf": "French_Guiana", "gg": "Guernsey", "gh": "Ghana", "gi": "Gibraltar",
    "gl": "Greenland", "gm": "Gambia", "gn": "Guinea", "gp": "Guadeloupe", "gq": "Equatorial_Guinea",
    "gr": "Greece", "gs": "South_Georgia", "gt": "Guatemala", "gu": "Guam", "gw": "Guinea-Bissau",
    "gy": "Guyana", "hk": "Hong_Kong", "hm": "Heard_Island", "hn": "Honduras", "hr": "Croatia",
    "ht": "Haiti", "hu": "Hungary", "id": "Indonesia", "ie": "Ireland", "il": "Israel",
    "im": "Isle_of_Man", "in": "India", "iq": "Iraq", "ir": "Iran", "is": "Iceland",
    "it": "Italy", "je": "Jersey", "jm": "Jamaica", "jo": "Jordan", "jp": "Japan",
    "ke": "Kenya", "kg": "Kyrgyzstan", "kh": "Cambodia", "ki": "Kiribati", "km": "Comoros",
    "kn": "Saint_Kitts_Nevis", "kp": "North_Korea", "kr": "South_Korea", "kw": "Kuwait", "ky": "Cayman_Islands",
    "kz": "Kazakhstan", "la": "Lao", "lb": "Lebanon", "lc": "Saint_Lucia", "li": "Liechtenstein",
    "lk": "Sri_Lanka", "lr": "Liberia", "ls": "Lesotho", "lt": "Lithuania", "lu": "Luxembourg",
    "lv": "Latvia", "ly": "Libya", "ma": "Morocco", "mc": "Monaco", "md": "Moldova",
    "me": "Montenegro", "mf": "Saint-Martin", "mg": "Madagascar", "mh": "Marshall_Islands ", "mk": "North_Macedonia",
    "ml": "Mali", "mm": "Myanmar", "mn": "Mongolia", "mo": "Macao", "mp": "Northern_Mariana_Islands",
    "mq": "Martinique", "mr": "Mauritania", "ms": "Montserrat", "mt": "Malta", "mu": "Mauritius",
    "mv": "Maldives", "mw": "Malawi", "mx": "Mexico", "my": "Malaysia", "mz": "Mozambique",
    "na": "Namibia", "nc": "New_Caledonia", "ne": "Niger", "nf": "Norfolk_Island", "ng": "Nigeria",
    "ni": "Nicaragua", "nl": "Netherlands", "no": "Norway", "np": "Nepal", "nr": "Nauru",
    "nu": "Niue", "nz": "New_Zealand", "om": "Oman", "pa": "Panama", "pe": "Peru",
    "pf": "French_Polynesia", "pg": "Papua_New_Guinea", "ph": "Philippines", "pk": "Pakistan", "pl": "Poland",
    "pm": "Saint_Pierre", "pn": "Pitcairn", "pr": "Puerto Rico", "ps": "Palestine", "pt": "Portugal",
    "pw": "Palau", "py": "Paraguay", "qa": "Qatar", "re": "Réunion", "ro": "Romania",
    "rs": "Serbia", "ru": "Russia", "rw": "Rwanda", "sa": "Saudi_Arabia", "sb": "Solomon_Islands",
    "sc": "Seychelles", "sd": "Sudan", "se": "Sweden", "sg": "Singapore", "sh": "Saint_Helena",
    "si": "Slovenia", "sk": "Slovakia", "sl": "Sierra_Leone", "sm": "San_Marino", "sn": "Senegal",
    "so": "Somalia", "sr": "Suriname", "ss": "South_Sudan", "st": "Sao_Tome", "sv": "El_Salvador",
    "sx": "Sint_Maarten", "sy": "Syria", "sz": "Eswatini", "tc": "Caicos_Islands", "td": "Chad",
    "tf": "French_Southern", "tg": "Togo", "th": "Thailand", "tj": "Tajikistan", "tk": "Tokelau",
    "tl": "Timor-Leste", "tm": "Turkmenistan", "tn": "Tunisia", "to": "Tonga", "tp": "East_Timor",
    "tr": "Turkey", "tt": "Trinidad_Tobago", "tv": "Tuvalu", "tw": "China", "tz": "Tanzania",
    "ua": "Ukraine", "ug": "Uganda", "uk": "United_Kingdom", "us": "United_States", "uy": "Uruguay",
    "uz": "Uzbekistan", "va": "The_Vatican", "vc": "Saint_Vincent", "ve": "Venezuela", "vg": "Virgin_Islands",
    "vi": "Virgin_Islands", "vn": "Viet_Nam", "vu": "Vanuatu", "wf": "Wallis_Futuna", "ws": "Samoa",
    "ye": "Yemen", "yt": "Mayotte", "za": "South_Africa", "zm": "Zambia", "zw": "Zimbabwe",
    "ελ": "Greece", "бг": "Bulgaria", "бел": "Bulgaria", "мкд": "North_Macedonia", "рф": "Russia",
    "срб": "Serbia", "укр": "Ukraine", "қаз": "Kazakhstan", "հայ": "Armenia", "الاردن": "Jordan",
    "الجزائر": "Algeria", "السعودية": "Saudi_Arabia", "المغرب": "Morocco", "امارات": "United_Arab_Emirates",
    "ایران": "Iran",
    "بھارت": "India", "تونس": "Tunisia", "سودان": "Sudan", "سورية": "Syria", "عراق": "Iraq",
    "عمان": "Oman", "فلسطين": "Palestine", "قطر": "Qatar", "مصر": "Egypt", "مليسيا": "Malaysia",
    "موريتانيا": "Mauritania", "پاكستان": "Pakistan", "پاکستان": "Pakistan", "ڀارت": "India", "भारत": "India",
    "বাংলা": "Bangladesh", "ভারত": "India", "ਭਾਰਤ": "India", "ભારત": "India", "இந்தியா": "India",
    "இலங்கை": "Sri_Lanka", "சிங்கப்பூர்": "Singapore", "భారత్": "India", "ಭಾರತ": "India", "ഭാരതം": "India",
    "ලංකා": "Thailand", "ไทย": "Thailand", "中国": "China", "中國": "China", "台湾": "Taiwan",
    "台灣": "Taiwan", "新加坡": "Singapore", "澳門": "Macao", "香港": "Hong_Kong", "한국": "South_Korea"
}

COUNTRY_CODES_REGION: Dict[str, str] = {
    "ad": "europe_west", "ae": "middle_east", "af": "asia_central", "al": "europe_west", "ao": "africa_sub",
    "aq": "antarctica", "ar": "america_south", "as": "asia_southeast",
    "at": "europe_west", "au": "oceania", "aw": "america_central", "ax": "europe_west", "az": "asia_central",
    "ba": "europe_east", "bb": "america_central", "bd": "asia_south",
    "be": "europe_west", "bf": "africa_sub", "bg": "europe_east", "bh": "middle_east", "bi": "africa_sub",
    "bj": "africa_sub", "bl": "american_central", "bm": "america_central",
    "bn": "asia_southeast", "bo": "america_south", "bq": "america_central", "br": "america_brazil",
    "bs": "america_central", "bt": "asia_south", "bv": "europe_west", "bw": "africa_southern",
    "by": "europe_east", "bz": "america_central", "ca": "america_north", "cd": "africa_sub", "cf": "africa_sub",
    "cg": "africa_sub", "ch": "europe_west", "ci": "africa_sub",
    "ck": "asia_southeast", "cl": "america_south", "cm": "africa_sub", "cn": "asia_east", "co": "america_south",
    "cr": "america_central", "cu": "america_central", "cv": "africa_sub",
    "cw": "america_central", "cx": "asia_southeast", "cy": "europe_west", "cz": "europe_east", "de": "europe_west",
    "dj": "africa_north", "dk": "europe_west", "dm": "america_central",
    "do": "america_central", "dz": "africa_north", "ec": "america_south", "ee": "europe_east", "eg": "middle_east",
    "er": "africa_north", "es": "europe_west", "et": "africa_north",
    "fi": "europe_west", "fj": "asia_southeast", "fk": "america_south", "fm": "asia_southeast", "fo": "europe_west",
    "fr": "europe_west", "ga": "africa_sub", "gb": "europe_west",
    "gd": "america_central", "ge": "asia_central", "gf": "america_south", "gh": "africa_sub", "gi": "africa_north",
    "gl": "europe_west", "gm": "africa_sub", "gn": "africa_sub",
    "gp": "america_central", "gr": "europe_west", "gt": "america_central", "gu": "oceania", "gw": "africa_sub",
    "gy": "america_south", "hk": "asia_east", "hn": "america_central",
    "hr": "europe_east", "ht": "america_central", "hu": "europe_east", "id": "asia_southeast", "ie": "europe_west",
    "il": "middle_east", "im": "europe_west", "in": "asia_south",
    "iq": "middle_east", "ir": "asia_central", "is": "europe_west", "it": "europe_west", "je": "europe_west",
    "jm": "america_central", "jo": "middle_east", "jp": "asia_east",
    "ke": "africa_sub", "kg": "asia_central", "kh": "asia_southeast", "ki": "asia_southeast", "km": "africa_sub",
    "kn": "america_central", "kp": "asia_east", "kr": "asia_east",
    "kw": "middle_east", "ky": "america_central", "kz": "asia_central", "lb": "middle_east", "lc": "america_central",
    "li": "europe_west", "lk": "asia_south", "lr": "africa_sub",
    "ls": "africa_southern", "lt": "europe_east", "lu": "europe_west", "lv": "europe_east", "ma": "africa_north",
    "mc": "europe_west", "md": "europe_east", "mf": "america_central",
    "mg": "africa_sub", "mh": "oceania", "mk": "europe_east", "ml": "africa_sub", "mm": "asia_southeast",
    "mn": "asia_east", "mo": "asia_east", "mp": "oceania",
    "mq": "america_central", "mr": "africa_sub", "mt": "europe_west", "mu": "asia_southeast", "mv": "europe_west",
    "mw": "africa_sub", "mx": "america_central", "my": "asia_southeast",
    "mz": "africa_sub", "na": "africa_southern", "nc": "oceania", "ne": "africa_sub", "nf": "oceania",
    "ng": "africa_sub", "ni": "america_central", "nl": "europe_west",
    "no": "europe_west", "np": "asia_south", "nr": "asia_southeast", "nz": "oceania", "om": "middle_east",
    "pa": "america_central", "pe": "america_south", "pf": "asia_southeast",
    "pg": "asia_southeast", "ph": "asia_southeast", "pk": "asia_south", "pl": "europe_east", "pm": "america_north",
    "pr": "america_central", "ps": "middle_east", "pt": "europe_west",
    "pw": "asia_southeast", "py": "america_south", "qa": "middle_east", "re": "africa_sub", "ro": "europe_east",
    "rs": "europe_east", "ru": "europe_russia", "rw": "africa_sub",
    "sa": "middle_east", "sb": "asia_southeast", "sc": "asia_south", "sd": "africa_north", "se": "europe_west",
    "sg": "asia_southeast", "si": "europe_east", "sk": "europe_east",
    "sl": "africa_sub", "sm": "asia_southeast", "sn": "africa_sub", "so": "africa_north", "sr": "america_south",
    "ss": "africa_sub", "su": "europe_russia", "sv": "america_central",
    "sx": "america_central", "sy": "middle_east", "sz": "africa_southern", "tc": "america_central", "td": "africa_sub",
    "tg": "africa_sub", "th": "asia_southeast", "tj": "asia_central",
    "tl": "asia_southeast", "tm": "asia_central", "tn": "africa_north", "tp": "asia_southeast", "tr": "middle_east",
    "tt": "america_central", "tw": "asia_east", "tz": "africa_sub",
    "ua": "europe_east", "ug": "africa_sub", "uk": "europe_west", "us": "america_north", "uy": "america_south",
    "uz": "asia_central", "va": "europe_west", "vc": "america_central",
    "ve": "america_south", "vg": "america_central", "vi": "america_central", "vn": "asia_southeast",
    "vu": "asia_southeast", "wf": "asia_southeast", "ye": "middle_east", "yt": "africa_sub",
    "za": "africa_southern", "zm": "africa_sub", "zw": "africa_southern", "ελ": "europe_west", "бг": "europe_east",
    "бел": "europe_east", "мкд": "europe_east", "рф": "europe_russia",
    "срб": "europe_east", "укр": "europe_east", "қаз": "asia_central", "հայ": "asia_central", "الاردن": "middle_east",
    "الجزائر": "africa_north", "السعودية": "middle_east", "المغرب": "middle_east",
    "امارات": "middle_east", "ایران": "middle_east", "بھارت": "asia_south", "تونس": "africa_north",
    "سودان": "africa_sub", "سورية": "middle_east", "عراق": "middle_east", "عمان": "middle_east",
    "فلسطين": "middle_east", "قطر": "middle_east", "مصر": "middle_east", "مليسيا": "asia_southeast",
    "موريتانيا": "africa_north", "پاكستان": "asia_south", "پاکستان": "asia_south", "ڀارت": "asia_south",
    "भारत": "asia_south", "বাংলা": "asia_south", "ভারত": "asia_south", "ਭਾਰਤ": "asia_south", "ભારત": "asia_south",
    "இந்தியா": "asia_south", "இலங்கை": "asia_south", "சிங்கப்பூர்": "asia_southeast",
    "భారత్": "asia_south", "ಭಾರತ": "asia_south", "ഭാരതം": "asia_south", "ලංකා": "asia_southeast",
    "ไทย": "asia_southeast", "中国": "asia_east", "中國": "asia_east", "台湾": "asia_east",
    "台灣": "asia_east", "新加坡": "asia_southeast", "澳門": "asia_east", "香港": "asia_east", "한국": "asia_east",
    "st": "africa_sub"
}
# remove the url that repeat the same pages in many countries
URL_FILTER = (
    'hotel', 'remax', 'britishcouncil', 'life', 'yellowpages', 'apteka', 'travel', '24', 'nic', 'gouv', 'job',
    'landrover',
    'meteo', 'bmw', 'ford', 'toyota', 'manpower', 'renault', 'citroen', 'peugeot', 'airfrance', 'europcar', 'nissan',
    'inet', 'kayak', 'avis', 'mercedes-benz', 'cosplayfu', 'pinterest', 'bachelorstudies', 'panasonic', 'masterstudies',
    'silvergoldbull', 'jll', 'ntv', 'habbo', 'academiccourses', 'sephora', 'gay', 'eva', 'tenstickers', 'culture',
    'monitor', 'thepiratebay', 'booked', 'destinia', 'blogspot', 'choowap', 'edel-optics', 'industrystock', 'piaget',
    'datasheet', 'chopard', 'tuugo', 'chrono24', 'digikey', 'visahq', 'allrecipes', 'redbull', 'webnode', 'focus',
    'today',
    'vessoft', 'huffingtonpost', 'molnlycke', 'radio', 'retsch', 'schaeffler', 'viagogo', 'volvotrucks', 'weather',
    'websitelibrary', 'jetradar', 'simplyhired', 'lyrics', 'metro', 'drivershere', 'bnb', 'catholic', 'encycolorpedia',
    'politics', 'projects-abroad', 'bmcsoftware', 'mytaste', 'ticmate', 'vita', 'icecat', 'comsol', 'skyscanner',
    'freemeteo', 'groupon', 'caritas', 'huno', 'sapo', 'vogue', 'abc', 'funnygames', 'transfermarkt', 'flagma',
    'gorenje',
    'indeed', 'lomography', 'tripadvisor', 'ebike-base', 'ibooked', 'olympic', 'freelancer', 'nobleprog', 'hotels',
    'poki',
    'spartoo', 'laredoute', 'europages', 'fishbase', 'nvidia', 'wimdu', 'amazon', 'mathworks', 'regus', 'ebay',
    'directferries', 'schneider-electric', 'talkreviews', 'fairmont', 'agriaffaires', 'solarmovie', 'eurocampings',
    'lookfantastic', 'expedia', 'google', 'radissonblu', 'intel', 'sandisk', 'gamereactor', 'airbnb', 'hotfrog',
    'staples',
    'mts', 'humanrights', 'sports', 'mama', 'hardware', 'catawiki', 'esprit', 'islam', 'audi', 'sport1', 'hh',
    'fjallraven',
    'weatheronline', 'wika', 'equinix', 'efinancialcareers', 'doublegames', 'randstad', 'wiggle', 'testfreaks',
    'babycenter', 'morningstar', 'lorealprofessionnel', 'domyos', 'play', 'brother', 'myprotein', 'instron',
    'villeroy-boch', 'linux', 'bosch-home', 'hager', 'longines', 'unicef', 'woxikon', 'hilti', 'fxpro', 'hipp', 'omron',
    'gifmania', 'mitula', 'sixt', 'leroymerlin', 'pepperl-fuchs', 'stihl', 'creativecommons', 'bose', 'meteoprog',
    'panorama', 'laitman', 'visitdenmark', 'ine', 'elle', 'kb', 'adidas', 'game-game', 'careerjet', 'jobisjob',
    'tribord',
    'yves-rocher', 'gettyimages', 'intersport', 'plus500', 'tui', 'holidaycheck', 'ucoz', 'autoeurope', 'denios',
    'vlex',
    'keuco', 'shopmania', 'songspk', 'government', 'putlocker', 'yp', 'monotaro', 'homeaway', 'tvtrip', 'nintendo',
    'minube', 'homify', 'stubhub', 'heidenhain', 'pwc', 'esquire', 'decathlon', 'intertool', 'iha', 'travelnews',
    'blog',
    'amnesty', 'momondo', 'wikimedia', 'golf', 'siemens', 'redcross', 'forum', 'orange', 'jooble', 'deliveroo',
    'volkswagen', 'handball', 'mfa', 'honda', 'gamers', 'cari', 'citywire', 'livescore', 'mp3skull', 'watchseries',
    'foodpanda', 'eventbrite', 'opendi', 'localmart', 'trivago', 'unesco', 'anunico', '99designs', 'laroche-posay',
    'genious', 'millesima', 'vichy', 'neuvoo', 'ubuntu', 'century21', 'pokerlistings', 'msf', 'mybb', 'sony', 'alatest',
    'cosmo', 'rtl', 'epson', 'adoos', 'viva', 'buro247', 'moviesonline', 'president', 'market', 'top', 'time', 'egov',
    'cosmopolitan', 'fragrantica', 'amcham', 'camping', 'transparency', 'atlascopco', 'money', 'nationalgeographic',
    'skoda', 'wwf', 'bauhaus', 'lastfm', 'ciao', 'canon', 'yasni', 'nestle', 'agoda', 'harpersbazaar', 'timeout',
    'institutfrancais', 'vistaprint', 'msccruises', 'trovit', 'fishersci', 'reebok', 'atea', 'music', 'capital', 'pomu',
    'cartoonnetwork', 'education', 'eurosport', 'grazia', 'army', 'mascus', 'dormeo', 'mir', 'click', 'gazeta', 'sport',
    'scientology', 'filetypes', 'fruugo', 'seton', 'autodesk', 'softpicks', 'sparta', 'merchandisingplaza', 'femina',
    'aquatuning', 'tiendeo', 'atv', 'businessinsider', 'top-shop', 'micro-epsilon', 'franchising', 'lexus', 'coface',
    'marieclaire', 'sputnik', 'posters', 'scandinaviandesigncenter', 'autosport', 'computerworld', 'tribuna', 'radio1',
    'posta', 'superprof', 'olx', 'expert', 'navabi', 'energia', 'autoline-eu', 'lookingforbooking', 'philips', 'vesti',
    'danubiushotels', 'realigro', 'woman', 'kia', 'rp5', 'obi', 'games', 'shop-rc-models', '2gis', 'tele2', 'museum',
    'glamour', 'kp', 'adriatic-home', 'auto', 'rabota', 'imaginarium', 'jetcost', 'kino', 'oferteo', 'softkey',
    'bimago',
    'snowtrex', 'nickelodeon', 'interhome', 'topdestination', 'forbes', 'flatfy', 'bonprix', 'esky', 'douglas',
    'michelin',
    'telekom', 'zooplus', 'picclick', 'makita', 'thebodyshop', 'open-closed', 'tv3', 'samsonite', 'conrad', 'donkiz',
    'amway', 'zankyou', 'doctoralia', 'electrolux', 'mediamarkt', 'lgblog', 'blogs', 'opel', 'nordea', 'dnevnik', 'if',
    'tchibo', 'nivea', 'standard', 'zeiss', 'gismeteo', 'fashion', 'aga', 'wolfcraft', 'bridgestone', 'opera', 'sgs',
    'shopalike', 'docplayer', 'index', 'klingel', 'mtv', 'zoover', 'machineryzone', 'home', 'dekoria', 'autoscout24',
    'kelkoo', 'westwing', 'flixbus', 'football', 'coop', 'download', 'webwiki', 'eventim', 'independent', 'justice',
    'turismoi', 'vuelosbaratos', 'geek', 'chamber', '1-urlm', 'clasf', 'divendo', 'cylex', 'autoline', 'jobtonic',
    'metin2',
    'edigital', 'volleyball', 'hbo', 'basket', 'reporter', 'mister-auto', 'shirtcity', 'anywayanyday', 'literatura',
    'novasol', 'berlitz', 'urlm', 'shell', 'fitshop', 'doplim', 'tirendo', 'emagister', 'slideplayer', 'telegraf',
    'stat',
    'spreadshirt', 'zazzle', 'toysrus', 'monster', 'flashscore', 'just-eat', 'allposters', 'forumgratuit', 'uni',
    'ticketpro', 'ticketmaster', 'pixum', 'michaelpage', 'jobs', 'nuroa', 'autoblog', 'vodafone', 'pinkorblue',
    'studentjob', 'home24', 'parlament', 'topshop', 'automobile', 'menshealth', 'ekosport', 'faktor', 'gohome',
    'bravofly',
    'pcworld', 'afribaba', 'jumia', 'nu3', 'logismarket', 'solostocks', 'sarenza', 'skiinfo', 'eurogamer', 'euronics',
    'europosters', 'sportsevents365', 'gear4music', 'viamichelin', 'zalando', 'campingcard', 'bax-shop', 'meteovista',
    'teloos', 'yachtworld', 'notino', 'bikester', 'fitnessdigital', 'euractiv', 'nded', 'transitcenter', 'happyhair',
    'killerinktattoo', 'openhours', '24mx', 'ecco-verde', 'onlineprinters', 'boels', 'casamundo', 'atraveo', 'mercateo',
    'thelocal', 'stepstone', 'bechtle', 'bridesire', 'rajapack', 'roefix', 'electronic-star', 'holidayguru',
    'rapunzelofsweden', 'kilroy', 'gombis', 'skatepro', 'yoursurprise', 'campz', 'peterhahn', 'telia'
)


def remove_emoji(text: str):
    return emoji.replace_emoji(text, replace='')


def extract_url(url: str):
    result = tldextract.extract(url)
    return result.domain, result.suffix.split('.')[-1]


def extract_n_grams(text: str, n: int = 1):
    return nltk.ngrams(text, n)


def get_url_filters_from_file(file_dir) -> Dict[str, Dict[str, int]]:
    """
    Getting dictionary of url filters from file
    e.g. {'hotel': {'num_of_countries': 9, 'num_of_pages': 253383},...}
    """
    df = pd.read_csv(file_dir,
                     sep=' ',
                     names=['domain', 'num_of_countries', 'num_of_pages'],
                     index_col='domain')
    return df.to_dict(orient='index')


def write_url_filters_to_file(file_dir, filters: Dict[str, Dict[str, int]]):
    """
    Writing to a space delimited file, with columns as follows
    Base url - Number of countries - Number of pages
    """
    with open(file_dir, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=' ')
        for key, value in filters.items():
            row = [key, value['num_of_countries'], value['num_of_pages']]
            writer.writerow(row)


def divide_list(input_list, chunk_size):
    length = len(input_list)
    # Use the zip function to split the input list into the desired number of sublists
    return [input_list[i:i + chunk_size] for i in range(0, length, chunk_size)]


def strip_tags(line: str) -> str:
    line = re.sub(r"http\S+", "", line)
    line = re.sub(r"@\S+", "", line)
    line = re.sub(r"#\S+", "", line)
    line = re.sub("<[^>]*>", "", line)
    return line
