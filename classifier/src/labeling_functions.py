#from analysis_text import keywords_lf_generator, regex_lf_generator
import re
from snorkel.labeling import labeling_function

# Constants
UKRAINE = 1
COVID = 3
CRISIS = 5
BREXIT = 6


ABSTAIN = -1

lfs_dict = dict()
collection_dicts = dict()

#### Labeling functions generators
def keywords_lf_generator(keywords: list, 
                          constant: int, 
                          name: str = 'keywordsLabelingFunction',
                          ABSTAIN: int = -1):
    @labeling_function()
    def func(x):
        return constant if any(word in x.text.lower() for word in keywords) else ABSTAIN
    func.name = name
    return func
        
def regex_lf_generator(regex, 
                       constant: int, 
                       name: str = 'regexLabelingFunction',
                       ABSTAIN: int = -1):
    @labeling_function()
    def func(x):
        return constant if re.search(regex, x.text.lower(), flags=re.I) else ABSTAIN
    func.name = name
    return func


######## Ukraine lfs
ukraine_keywords = [
                'ukraine', 'russia', 'ukrainian', 'russian', 'putin', "zelensky", 'zelensk', "mariupol", 
                "luhansk", "donetsk", "#ukraine", 'donbas', 'invasion', 'finland', 'oligarch',
                'visa', 'nato', " war ", "ukraine family scheme"
                ]

ukraine_lfs = {
    'ukraine_keywords': keywords_lf_generator(ukraine_keywords, UKRAINE, 'ukraine_keywords'),
    'ukraine_regex': regex_lf_generator(r"ukrain.", UKRAINE, 'ukraine_regex'),
    'russia_regex': regex_lf_generator(r"russia", UKRAINE, 'russia_regex'),
    'putin_regex': regex_lf_generator(r"putin", UKRAINE, 'putin_regex'),
    'oligarch_regex': regex_lf_generator(r"oligarch", UKRAINE, 'oligarch_regex'),
    }

collection_dicts['ukraine_lfs'] = ukraine_lfs




######## COVID lfs
covid_keywords = [
                'covid', 'covid-19', 'sars-cov-2', 'coronavirus', 'pandemic', 'lockdown', 'vaccine', 'vaccines', 
                'rollout', 'vaccination', 
                #'cases'
                ]

covid_lfs = {
    'covid_keywords': keywords_lf_generator(covid_keywords, COVID, 'covid_keywords'),
    'vaccines_regex': regex_lf_generator(r"vaccin", COVID, 'vaccines_regex'),
    'rollout_regex': regex_lf_generator(r"vaccin\W+ rollout", COVID, 'rollout_regex'),
    'covid_regex': regex_lf_generator(r"covid", COVID, 'covid_regex'),
    'lockdown_regex': regex_lf_generator(r"lockdown", COVID, 'lockdown_regex'),
    'pandemic_regex': regex_lf_generator(r"vaccin", COVID, 'pandemic_regex'),   
    'corona_regex': regex_lf_generator(r"corona", COVID, 'corona_regex'),
}

collection_dicts['covid_lfs'] = covid_lfs



######## Cost of living crisis lfs
crisis_keywords = [
                "fuel", "bills", "petrol", "energy", "inflation", 'costs', 'electricity', 'oil',
                'poverty', 'austerity', 'windfall', 'poverty', 'cost-of-living', 'council tax'
                ]

crisis_lfs = {
    'crisis_keywords': keywords_lf_generator(crisis_keywords, CRISIS, 'crisis_keywords'),
    'living_regex': regex_lf_generator(r"cost\W+of\W+living", CRISIS, 'living_regex'),
    'living2_regex': regex_lf_generator(r"costofliving", CRISIS, 'living2_regex'),
    'prices_regex': regex_lf_generator(r"(increasing|rising|energy) (price|bill)", CRISIS, 'prices_regex'),
    'tax_regex': regex_lf_generator(r"tax", CRISIS, 'tax_regex'),
    'natins_regex': regex_lf_generator(r"national insurance", CRISIS, 'natins_regex'),
    'council_regex': regex_lf_generator(r"council tax", CRISIS, 'council_regex'),
    'inflation_regex': regex_lf_generator(r"(rising|increasing|historic|high) inflation", CRISIS, 'inflation_regex')
    }

collection_dicts['crisis_lfs'] = crisis_lfs

######## Brexit lfs
brexit_keywords = [
                'brexit', 'protocol', 'article 50', 'european union', 'northern ireland protocol', 
                #' eu '
                ]

brexit_lfs = {
    'brexit_keywords': keywords_lf_generator(brexit_keywords, BREXIT, 'brexit_keywords'),
    'brexit_regex': regex_lf_generator(r"brexit", BREXIT, 'brexit_regex'),
    'protocol_regex': regex_lf_generator(r"northern ireland protocol", BREXIT, 'protocol_regex'),
    'ni_regex': regex_lf_generator(r"northern ireland", BREXIT, 'ni_regex'),
    'goodfriday_regex': regex_lf_generator(r"good friday", BREXIT, 'goodfriday_regex'),
    'article50_regex': regex_lf_generator(r"article 50", BREXIT, 'article50_regex'),   
    'eu_regex': regex_lf_generator(r"european union", BREXIT, 'eu_regex'),
}

collection_dicts['brexit_lfs'] = brexit_lfs


######### Export dictionary

for (_, dict_) in collection_dicts.items():
    for (key, lf) in dict_.items():
        lfs_dict[key] = lf

