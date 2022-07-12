from glob import glob
from os.path import basename, join
from pathlib import Path
import re

from lxml import etree

NS = "{http://www.tei-c.org/ns/1.0}"

# yanked from https://github.com/tnhaider/xml-poetry-reader/blob/master/utils/helper.py
def get_versification(meter_line, measure_type='f', greek_forms=True):
        # full = f
        # short = s
        # intermediate = i
        meter = ''.join(meter_line)
        meter = re.sub(r'\+', 'I', meter)
        meter = re.sub(r'\-', 'o', meter)
        hexameter =       re.compile('^Ioo?Ioo?Ioo?Ioo?IooIo$')
        alxiambichexa =   re.compile("^oIoIoIoIoIoIo?$")
        asklepiade =      re.compile("^IoIooIIooIoI$") # 12 Ode
        glykoneus =       re.compile("^IoIooIoI$")     # 8  Ode
        pherekrateus =    re.compile("^IoIooIo$")      # 7  Ode
        iambelegus =      re.compile('^oIoIoIoIIooIooI$')
        elegiambus =      re.compile('^IooIooIo?oIoIoIoo?$')
        diphilius =       re.compile('^IooIooI..IooI..$')
        prosodiakos =     re.compile('^.IooIooII?$')
        sapphicusmaior =  re.compile('^IoIIIooIIooIoI.$')
        sapphicusminor =  re.compile('^IoI.IooIoI.$')
        iambicseptaplus = re.compile("^oIoIoIoIoIoIoIo?")
        iambicpenta =     re.compile("^oIoIoIoIoIo?$")
        iambicpentaspond= re.compile("^IIoIoIoIoIo?$")
        iambictetra =     re.compile("^.IoIoIoIo?$")
        iambictri =       re.compile("^.IoIoIo?$")
        iambicdi =        re.compile("^.IoIo?$")
        iambic =          re.compile("^.IoIo?")
        iambicsingle =    re.compile("^oI$")
        trochaicseptaplus =  re.compile('^IoIoIoIoIoIoIo?')
        trochaichexa =       re.compile('^IoIoIoIoIoIo?$')
        trochaicpenta =      re.compile('^IoIoIoIoIo?$')
        trochaictetra =      re.compile('^IoIoIoIo?$')
        trochaictri =        re.compile('^IoIoIo?$')
        trochaicdi =         re.compile('^IoIo?$')
        trochaicsingle =     re.compile("^Io$")
        trochaic =           re.compile('^IoIo?')
        amphibrachdi =         re.compile('^o?IooIo$')
        amphibrachdimix =      re.compile('^oIooIo')
        amphibrachtri =        re.compile('^oIooIooIo?$')
        amphibrachtriplus =    re.compile('^oIooIooIo')
        amphibrachtetra =      re.compile('^oIooIooIooIo?$')
        amphibrachtetraplus =  re.compile('^oIooIooIooIo')
        amphibrachpentaplus =  re.compile('^oIooIooIooIooIo?')
        amphibrachsingle =     re.compile('^oIo$')
        adoneus =         re.compile('^IooI.$')
        adoneusspond =    re.compile('^IooII$')
        dactylicpenta =     re.compile('^IooIooIooIooIo?o?$')
        dactylicpentaplus = re.compile('^IooIooIooIooIooIoo')
        dactylictetra =     re.compile('^IooIooIooIo?o?$')
        dactylictetraplus = re.compile('^IooIooIooIoo')
        dactylictri =       re.compile('^IooIooIo?o?$')
        dactylictriplus =   re.compile('^IooIooIoo')
        dactylicdi =        re.compile('^IooIoo$')
        dactylicdiplus =    re.compile('^IooIoo')
        amphibrachiambicmix =  re.compile('^oI.*oIooIoo?I')
        amphibrachtrochaicmix =   re.compile('^Io.*oIooIoo?I')
        artemajor =       re.compile('^oIooIooIooIo$')
        artemajorhalf =   re.compile('^oIooIo$')
        iambicseptainvert=re.compile("^IooIoIoIoIoIoIo?$")
        iambichexainvert =re.compile("^IooIoIoIoIoIo?$")
        iambicpentainvert=re.compile("^IooIoIoIoIo?$")
        iambictetrainvert=re.compile("^IooIoIoIo?$")
        iambictriinvert = re.compile("^IooIoIo?$")
        iambicinvert =    re.compile('^IooIoI')
        trochaicextrasyll=   re.compile('^I.*IooI.+')
        iambicextrasyll=  re.compile('^o.*IooI.+')
        #iambiccholstrict =re.compile('.IoI.IoIoII.$')
        iambiccholstrict =re.compile("^oIoIoIoIoIooI$")
        iambicchol       =re.compile('^o?.*IooI$')
        zehnsilber =      re.compile('^...I.....I$')
        anapaestdiplus =  re.compile('^ooIooI')
        anapaesttriplus = re.compile('^ooIooIooI')
        anapaesttetraplus=re.compile('^ooIooIooIooI')
        anapaestinit =    re.compile('^ooI')
        dactylicinit =      re.compile('^o?Ioo')
        spondeus =        re.compile('^II$')
        singleup =        re.compile('^I$')
        singledown =      re.compile('^o$')
        #alexandriner =    re.compile('oIoIoIoIoIoIo?$')
        #adoneus =        re.compile('IooIo$')
        #iambicamphibrachcentermix = re.compile('oIoIooIoI$')

        greek = { 'asklepiade':asklepiade,\
                  'glykoneus':glykoneus,\
                  'pherekrateus':pherekrateus,\
                  'iambelegus':iambelegus,\
                  'elegiambus':elegiambus,\
                  'diphilius':diphilius,\
                  'prosodiakos':prosodiakos,\
                  'sapphicusmaior':sapphicusmaior,\
                  'sapphicusminor':sapphicusminor
                 }

        adoneus = { 
                  'adoneus':adoneus,\
                  'adoneus.spond':adoneusspond
                  }

        verses1 = {'iambic.septa.plus':iambicseptaplus,\
                  'hexameter':hexameter,\
                  'alexandrine.iambic.hexa':alxiambichexa,\
                  'iambic.penta':iambicpenta,\
                  'iambic.penta.spondeus':iambicpentaspond,\
                  'iambic.tetra':iambictetra,\
                  'iambic.tri':iambictri,\
                  'iambic.di':iambicdi,\
                  'trochaic.septa.plus':trochaicseptaplus,\
                  'trochaic.hexa':trochaichexa,\
                  'trochaic.penta':trochaicpenta,\
                  'trochaic.tetra':trochaictetra,\
                  'trochaic.tri':trochaictri,\
                  'trochaic.di':trochaicdi
                  }

        verses2 = {
                  'dactylic.penta':dactylicpenta,\
                  'dactylic.tetra':dactylictetra,\
                  'dactylic.tri':dactylictri,\
                  'amphibrach.penta.plus':amphibrachpentaplus,\
                  'amphibrach.tetra':amphibrachtetra,\
                  'amphibrach.tetra.plus':amphibrachtetraplus,\
                  'amphibrach.tri':amphibrachtri,\
                  'amphibrach.tri.plus':amphibrachtriplus,\
                  'amphibrach.relaxed':amphibrachdi,\
                  'dactylic.penta.plus':dactylicpentaplus,\
                  'dactylic.tetra.plus':dactylictetraplus,\
                  'dactylic.tri.plus':dactylictriplus,\
                  'dactylic.di.plus':dactylicdiplus,\
                  'dactylic.di':dactylicdi,\
                  'anapaest.tetra.plus':anapaesttetraplus,\
                  'anapaest.tri.plus':anapaesttriplus,\
                  'anapaest.di.plus':anapaestdiplus,\
                  'arte_major':artemajor,\
                  'arte_major.half':artemajorhalf
                  }

        verses3 = {
                  'iambic.septa.invert':iambicseptainvert,\
                  'iambic.hexa.invert':iambichexainvert,\
                  'iambic.penta.invert':iambicpentainvert,\
                  'iambic.tetra.invert':iambictetrainvert,\
                  'iambic.tri.invert':iambictriinvert,\
                  'iambic.invert':iambicinvert,\
                  'trochaic.relaxed':trochaicextrasyll,\
                  'iambic.relaxed':iambicextrasyll,\
                  'iambic.chol.strict':iambiccholstrict,\
                  'iambic.relaxed.chol':iambicchol,\
                  'amphibrach.single':amphibrachsingle,\
                  'amphibrach.iambic.mix':amphibrachiambicmix,\
                  'amphibrach.trochaic.mix':amphibrachtrochaicmix,\
                  'anapaest.init':anapaestinit,\
                  'dactylic.init':dactylicinit,\
                  'amphibrach.di.mix':amphibrachdimix,\
                  'zehnsilber':zehnsilber,\
                  'spondeus':spondeus,\
                  'iambic.single':iambicsingle,\
                  'trochaic.single':trochaicsingle,\
                  'single.down':singledown,\
                  'single.up':singleup}

        verses = {}
        if greek_forms == False:
                #verses = verses1 + verses2 + verses3
                verses.update(verses1)
                verses.update(verses2)
                verses.update(verses3)
        if greek_forms == True:
                verses.update(verses1)
                verses.update(greek)
                verses.update(verses2)
                verses.update(adoneus)
                verses.update(verses3)
                #verses = verses1 + greek + verses2 + adoneus + verses3

        label = None
        for label, pattern in verses.items():
                result = pattern.match(meter)
                #if label == 'chol.iamb':
                #       result = pattern.search(meter)
                hebungen = meter.count('I')
                counters = {0:'zero', 1:'single', 2:'di', 3:'tri', 4:'tetra', 5:'penta', 6:'hexa', 7:'septa'}
                if hebungen > 6:
                        hebungen_label = 'septa.plus'
                else:
                        hebungen_label = counters[hebungen]
                if 'relaxed' in label:
                        label = re.sub('.relaxed', '.' + hebungen_label + '.relaxed', label)
                if 'iambic.invert' in label:
                        label = re.sub('.invert', '.' + hebungen_label + '.invert', label)
                if result != None:
                        split = label.split('.')
                        if measure_type == 's':
                                return split[0]
                        if measure_type == 'i':
                                return '.'.join(split[:2])
                        else:
                                return label
        else: return 'other'

# based on https://github.com/tnhaider/metrical-tagging-in-the-wild/blob/main/data/English/SmallGold/get_annotation_forbetter.py
def align_syllables(footmeter):
    try:
        meter = re.sub(r'\|', '', footmeter)
        meter = re.sub('s', '+', meter)
        meter = re.sub('w', '-', meter)
        meter = re.sub(r'\^', '-', meter)
        versemeasure = get_versification(meter)
        if "." in versemeasure:
            foot, metre = versemeasure.split('.', 1)
            return foot, metre
        return versemeasure, versemeasure
    except TypeError:
        return None, None

# based from https://github.com/linhd-postdata/averell/blob/develop/src/averell/readers/forbetter4verse.py
def _parse_xml(xml_file):
    """Poem parser for 'For better for verse' corpus.
    We read the data and find elements like title, author, year, etc. Then
    we iterate over the poem text and we look for each stanza and  line data.
    :param xml_file: Path for the poem xml file
    :return: Dict with the data obtained from the poem
    :rtype: dict
    """
    poem = {}
    tree = etree.parse(str(xml_file))
    corpus_name = xml_file.parts[-4]
    root = tree.getroot()
    nsmap = root.nsmap
    tei_url = NS.replace("{", "").replace("}", "")
    if not any(ns == tei_url for ns in nsmap.values()):
        # no TEI declaration in input file, cannot parse file
        return
    manually_checked = False
    title = root.find(f".//{NS}title").text
    author = root.find(f".//{NS}author")
    author_text = "unknown"
    if author is not None:
        author_text = author.text
    date = root.find(f".//{NS}date")
    year = None
    if date is not None:
        year = root.find(f".//{NS}date").text
    poem.update({
        "poem_title": title,
        "author": author_text,
        "year": year
    })
    line_group_list = root.findall(f".//{NS}lg")
    stanza_list = []
    line_number = 0
    for stanza_number, line_group in enumerate(line_group_list):
        stanza_type = line_group.get("type")
        rhyme = line_group.get("rhyme")
        line_list = []
        stanza_text = []
        for line in line_group.findall(f"{NS}l"):
            line_dict = {}
            line_length = None
            met = line.get("met")
            foot, metre = align_syllables(met)
            seg_list = [re.sub(r"[\n ]+", " ", seg.xpath("string()")) for seg in
                        line.findall(f"{NS}seg")]
            line_text = "".join(seg_list)
            line_dict.update({
                "line_number": line_number + 1,
                "line_text": line_text,
                "metrical_pattern": met,
                "line_length": line_length,
                "foot": foot,
                "metre": metre,
            })
            line_list.append(line_dict)
            stanza_text.append(line_text)
            line_number += 1
        stanza_list.append({
            "stanza_number": stanza_number + 1,
            "stanza_type": stanza_type,
            "rhyme": rhyme,
            "lines": line_list,
            "stanza_text": "\n".join(stanza_text),
        })
    poem.update({
        "manually_checked": manually_checked,
        "stanzas": stanza_list,
        "corpus": corpus_name,
    })
    return poem


def _get_features(path):
    """Function to find each poem file and parse it
    :param path: Corpus Path
    :return: List of poem dicts
    :rtype: list
    """

    ecpa_folder = basename(glob(join(path, "for_better_for_verse-*"))[0])
    path = Path(path)
    xml_folders = [
        path / ecpa_folder / "poems",
        #path / ecpa_folder / "poems2" # seem to be only repetitions
    ]
    for folder in xml_folders:
        for filename in folder.rglob("*.xml"):
            result = _parse_xml(filename)
            if result is not None:
                yield result

def fbfv_loader(path, _):
    schemes = set()
    for i, poem in enumerate(_get_features(path)):
        #print(poem)
        #input()
        for stanza in poem['stanzas']:
            if stanza['rhyme']:
                if len(stanza['lines']) >= 4:
                    schemes.add((len(stanza['rhyme']), len(stanza['lines'])))
                    #if len(stanza['rhyme']) == 6 and  len(stanza['stanza_text'].split("\n")) == 1:
                    #    print(stanza)
                    #    input()
                    #if len(stanza['rhyme']) == 3 and stanza['lines'][-1]['line_number'] == 7:
                    #    print(stanza)
                    #    input()
    print(schemes)
    input()
