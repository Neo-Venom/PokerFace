"""Static pan-India clinical vocabulary that does not depend on the input having a clean/ folder.

Everything here is public, generic vocabulary (INN drug names, common Indian brand names, eponyms,
clinical abbreviations, hospital role words).  Nothing is taken from any evaluation dataset.

Used by the medical guard (what must survive) and the name detector (what is never a person's name).
"""
from __future__ import annotations

import re

# --------------------------------------------------------------------------- drugs (generic + Indian brands)
DRUG_NAMES = frozenset("""
paracetamol acetaminophen dolo crocin calpol combiflam brufen ibuprofen diclofenac voveran aceclofenac zerodol
nimesulide naproxen ketorolac tramadol ultracet morphine fentanyl pethidine buprenorphine tapentadol
amoxicillin moxikind augmentin clavulanate clavam azithromycin azithral azee zithromax cefixime zifi taxim
ceftriaxone monocef cefuroxime ceftum cefpodoxime cefoperazone ceftazidime cefepime cefazolin cephalexin
ciprofloxacin ciplox ofloxacin zanocin oflox levofloxacin levoflox levomac norfloxacin norflox moxifloxacin
metronidazole flagyl metrogyl tinidazole ornidazole doxycycline doxt minocycline clindamycin linezolid
vancomycin teicoplanin meropenem imipenem ertapenem piperacillin tazobactam piptaz amikacin gentamicin
tobramycin streptomycin colistin polymyxin nitrofurantoin fosfomycin cotrimoxazole septran bactrim
artesunate artemether lumefantrine chloroquine hydroxychloroquine hcqs primaquine quinine lariago
rifampicin isoniazid pyrazinamide ethambutol bedaquiline delamanid akt
pantoprazole pantop pantocid omeprazole omez rabeprazole razo esomeprazole nexpro lansoprazole ranitidine
rantac zantac famotidine sucralfate digene gelusil mucaine domperidone domstal ondansetron ondem emeset
vomikind metoclopramide perinorm itopride levosulpiride drotaverine drotin dicyclomine cyclopam meftal
hyoscine buscopan simethicone lactulose duphalac cremaffin bisacodyl dulcolax loperamide racecadotril
rifaximin ursodeoxycholic udiliv silymarin liv
amlodipine amlokind amlong stamlo cilnidipine cilacar nifedipine depin felodipine telmisartan telma
telmikind losartan losar losacar olmesartan valsartan irbesartan ramipril cardace enalapril envas lisinopril
perindopril atenolol aten metoprolol betaloc metolar bisoprolol concor nebivolol nebicard carvedilol propranolol
ciplar labetalol prazosin minipress clonidine methyldopa hydralazine furosemide lasix torsemide dytor
spironolactone aldactone hydrochlorothiazide chlorthalidone indapamide atorvastatin atorva tonact
rosuvastatin rosuvas rozavel crestor simvastatin fenofibrate ezetimibe aspirin ecosprin disprin loprin
clopidogrel clopilet clavix clopivas plavix prasugrel ticagrelor brilinta warfarin acitrom heparin
enoxaparin clexane fondaparinux rivaroxaban xarelto apixaban dabigatran streptokinase tenecteplase alteplase
nitroglycerin sorbitrate isosorbide nicorandil ranolazine ivabradine trimetazidine digoxin lanoxin amiodarone
adenosine diltiazem verapamil
metformin glycomet glucophage glimepiride amaryl gliclazide glizid glipizide sitagliptin januvia vildagliptin
galvus teneligliptin linagliptin empagliflozin jardiance dapagliflozin forxiga canagliflozin pioglitazone
voglibose acarbose insulin lantus glargine huminsulin mixtard novorapid actrapid humalog tresiba
levothyroxine thyroxine thyronorm eltroxin carbimazole neomercazole methimazole propylthiouracil
prednisolone wysolone omnacortil methylprednisolone medrol hydrocortisone dexamethasone dexona decadron
betamethasone betnesol fludrocortisone deflazacort
salbutamol asthalin levosalbutamol budesonide budecort formoterol foracort salmeterol seroflo fluticasone
montelukast montair theophylline deriphyllin doxofylline ipratropium duolin tiotropium tiova
ferrous ferric folic folvite autrin livogen orofer fefol shelcal cipcal calcimax calcium cholecalciferol
methylcobalamin neurobion becosules zincovit multivitamin
gabapentin gabapin pregabalin pregeb amitriptyline nortriptyline duloxetine levetiracetam levipil phenytoin
eptoin valproate valparin sodium carbamazepine tegretol oxcarbazepine lacosamide lamotrigine topiramate
clobazam lorazepam ativan clonazepam clonotril lonazep alprazolam alprax restyl diazepam valium midazolam
escitalopram nexito sertraline fluoxetine paroxetine venlafaxine mirtazapine olanzapine risperidone
quetiapine haloperidol aripiprazole lithium donepezil memantine levodopa carbidopa syndopa pramipexole
ropinirole trihexyphenidyl pacitane
cetirizine cetzine okacet levocetirizine levocet fexofenadine allegra loratadine desloratadine
chlorpheniramine pheniramine avil hydroxyzine atarax promethazine phenergan
albendazole zentel mebendazole ivermectin fluconazole forcan itraconazole voriconazole amphotericin
clotrimazole candid terbinafine acyclovir valacyclovir oseltamivir tamiflu remdesivir tenofovir
dolutegravir efavirenz zidovudine lamivudine favipiravir
tamsulosin urimax finasteride dutasteride sildenafil tadalafil allopurinol zyloric febuxostat colchicine
methotrexate azathioprine cyclosporine tacrolimus mycophenolate sulfasalazine leflunomide
erythropoietin oxytocin misoprostol mifepristone dinoprostone tranexamic etamsylate botroclot
vitamin ors electral saline dextrose ringer mannitol albumin bicarbonate gluconate
adrenaline epinephrine noradrenaline norepinephrine dopamine dobutamine vasopressin atropine
lignocaine lidocaine xylocaine bupivacaine propofol ketamine thiopentone succinylcholine vecuronium
neostigmine naloxone flumazenil
betadine povidone soframycin neosporin mupirocin fusidic silverex calamine permethrin
sinarest wikoryl cheston ascoril grilinctus benadryl dextromethorphan ambroxol guaifenesin bromhexine
sporlac econorm enterogermina racecadotril
""".split())

# --------------------------------------------------------------------------- INN / clinical morphology
# Suffixes that only occur in drug names or medical terms (names like Sukhvir / Ranvir are protected by the
# narrow antiviral suffix list; "April" is excluded explicitly).
_CLINICAL_SUFFIX_RE = re.compile(
    r"(?:mycin|micin|cillin|cycline|floxacin|oxacin|conazole|prazole|bendazole|tidine|sartan|statin|olol|dipine|"
    r"gliptin|gliflozin|formin|glinide|parin|xaban|gatran|semide|thiazide|setron|pramide|lukast|sonide|"
    r"olone|asone|zepam|zolam|triptan|profen|fenac|coxib|phylline|azosin|tropium|platin|taxel|rubicin|penem|"
    r"clovir|mivir|desivir|fovir|navir|gravir|buvir|tasvir|pasvir|ximab|zumab|tinib|mectin|gabalin|"
    r"itis|osis|aemia|emia|algia|ectomy|otomy|ostomy|plasty|scopy|graphy|penia|cytosis|megaly|"
    r"rrhoea|rrhea|plegia|paresis|trophy|sclerosis|carcinoma|sarcoma|lymphoma|blastoma|"
    # "-pathy" / "-uria" only with a medical stem: Ganapathy, Pasupathy, Bhuria are names
    r"(?:neuro|myo|retino|nephro|encephalo|entero|arthro|adeno|radiculo|vasculo|cardio|osteo)pathy|"
    r"(?:dys|haemat|hemat|poly|protein|olig|an|glycos|py|albumin|noct|bacteri|chyl)uria)$"
)
_SUFFIX_MIN_LEN = 6
_SUFFIX_EXCLUDE = frozenset({"april"})

# --------------------------------------------------------------------------- eponyms
# Eponym roots that are never Indian person names: clinical on their own.
STRONG_EPONYMS = frozenset("""
addison parkinson alzheimer hodgkin cushing crohn hashimoto kawasaki raynaud meniere huntington ewing burkitt
whipple guillain babinski romberg ryle foley apgar koch hansen marfan sjogren tourette asperger kaposi
klinefelter duchenne charcot behcet takayasu wegener goodpasture reiter zollinger sheehan bartter fanconi
hirschsprung meckel reye gaucher niemann pompe fabry hurler patau noonan prader angelman ebstein fallot
eisenmenger brugada prinzmetal dressler buerger peutz crigler dubin alagille kasai billroth nissen mcburney
kernig brudzinski trousseau chvostek kussmaul cheyne virchow colles monteggia galeazzi osgood schlatter
perthes quervain dupuytren boerhaave menetrier schatzki zenker ringer fournier wernicke korsakoff klumpke
tinel phalen lachman mcmurray trendelenburg rovsing courvoisier janeway koplik nikolsky auspitz gottron
heberden lewy creutzfeldt friedreich plummer riedel wilms pott paget glasgow mantoux widal valsalva
coombs holter snellen rinne bartholin wharton stensen eustachian fallopian purkinje langerhans
graafian sertoli leydig wenckebach mobitz kayser fleischer krukenberg pancoast sturge leriche wolff
moro stokes cheyne quincke corrigan duroziez traube hutchinson argyll ishihara schilling chediak wiskott
digeorge bruton hartnup kallmann waldenstrom wallenberg volkmann sprengel scheuermann
""".split())

# Eponyms that are also ordinary given names / surnames / English words: clinical ONLY in an eponym frame
# ("Wilson's disease", "Turner syndrome", "Bell's palsy"), never on their own ("Wilson Varghese").
NAMELIKE_EPONYMS = frozenset("""
wilson bell down turner graves murphy gilbert williams edwards jones hunter becker gardner lynch berger
barton bennett baker still conn cullen barrett homans osler ludwig erb hartmann horner lambert eaton mallory
weiss stevens johnson rotor alport ladd heller christmas lyme laennec levine kehr hamman battle bright broca
brodie bowen braxton hicks brenner caplan carney castleman chagas evans felty frey fuchs job meigs mirizzi
morgagni munchausen naegele negri pick ramsay russell sydenham usher weber weil willis smith jefferson
""".split())

EPONYM_HEAD_NOUNS = (
    "disease|diseases|syndrome|sign|signs|palsy|test|tube|catheter|lactate|solution|angina|gangrene|cyst|"
    "encephalopathy|procedure|operation|fracture|phenomenon|triad|pentad|reflex|node|nodes|nodule|nodules|spots|"
    "lesion|lesions|ulcer|esophagus|oesophagus|diverticulum|tumour|tumor|sarcoma|lymphoma|contracture|point|"
    "maneuver|manoeuvre|incision|score|scale|thyroiditis|arteritis|chorea|ataxia|dystrophy|anaemia|anemia|"
    "respiration|breathing|pulse|murmur|gait|hernia|repair|disorder|deformity|malformation|tear|ring|bodies|"
    "body|cell|cells|duct|gland|area|block|criteria|classification|crisis|purpura|granulomatosis|cirrhosis|"
    "dementia|neuralgia|neuroma|position|law|rule|chart|index|attack|aphasia|paralysis"
)
# Possessive frames where an unknown capitalised word is an eponym, not a person ("Fournier's gangrene").
# Kept to heads that are never said of a patient ("Ramu's fracture", "Asha's test report", "X's operation" stay names);
# eponym tests / tubes / solutions are covered by the explicit eponym lists instead.
EPONYM_STRONG_HEADS = (
    "disease|syndrome|sign|palsy|angina|gangrene|encephalopathy|phenomenon|triad|pentad|reflex|spots|esophagus|"
    "oesophagus|diverticulum|sarcoma|lymphoma|contracture|thyroiditis|arteritis|chorea|ataxia|dystrophy|"
    "respiration|maneuver|manoeuvre|malformation|purpura|granulomatosis|neuralgia|aphasia"
)

# --------------------------------------------------------------------------- abbreviations / ward vocabulary
CLINICAL_ABBREVIATIONS = frozenset("""
anm aww gnm mpw cho lhv ayush opd ipd icu nicu picu ccu micu sicu hdu ot er ed bp hr pr rr spo2 temp
wt ht bmi rbs fbs ppbs cbc hb hba1c tsh t3 t4 lft kft rft ecg ekg usg ct mri cxr esr crp inr aptt att dots
ors iv im sc po od bd tds qid hs sos prn stat npo nad wnl kco dm dm2 t2dm htn cad ckd copd tb uti uri lrti
urti acs cva mi hf af dvt aki cld gerd pud ibd sle ra oa bph pcos gdm pih lscs turp cabg ptca pci hiv
hbsag hcv vdrl ns1 igg igm loinc icd snomed fhir abdm ndhm pmjay mbbs md ms mch dnb bams bhms frcs mrcp
ncct cect fnac hpe pft eeg emg ncv afb cbnaat mlc dama gcs avpu ivf rl ns dns cpap bipap hfnc
nrbm ett ryles foleys ngt picc cvp abg vbg ttt
""".split())

# Hospital roles and relation nouns: describe a person, are never that person's name.
ROLE_WORDS = frozenset("""
patient patients pt pts attendant attendants relative relatives guardian caregiver caretaker bystander escort
visitor visitors informant nominee kin nok family relation wife husband son daughter mother father brother
sister sisters nephew niece uncle aunt aunty auntie grandson granddaughter grandfather grandmother grandchild
daughter-in-law son-in-law father-in-law mother-in-law brother-in-law sister-in-law spouse child baby infant
newborn neonate boy girl man woman lady gentleman male female beta beti bete betiyan pati patni bhai behen
bahen bhabhi devar sasur saas chacha chachi mama mami mausi nana nani dada dadi bua tau tai mummy mumma mummi
papa maa amma ammi abba nurse nurses staff sister-in-charge matron ward-boy wardboy aya ayah sweeper cleaner
guard security watchman driver pilot technician technologist pharmacist dietician dietitian physio
physiotherapist counsellor counselor clerk receptionist cashier manager supervisor coordinator incharge
in-charge volunteer worker workers anm doctor doctors dr consultant resident intern registrar surgeon
physician anaesthetist anesthetist radiologist pathologist paediatrician pediatrician gynaecologist
obstetrician cardiologist neurologist nephrologist oncologist psychiatrist dentist hod superintendent
owner caller sender receiver holder applicant claimant beneficiary member insured proposer employee gharwale
gharwalon gharwala family-members neighbour neighbor friend colleague
""".split())

# Title-case words that front scheme / programme / facility names, never people
NON_NAME_TITLE_WORDS = frozenset("""
yojana yojna abhiyan abhiyaan swasthya arogya aarogya rashtriya rashtriy national state district government
govt sarkari mission scheme card health insurance policy bima kendriya samiti sansthan trust charitable
foundation society municipal corporation panchayat block tehsil taluk zila zilla division primary community
centre center wellness anganwadi aanganwadi dispensary polyclinic medical college institute hospital
hospitals clinic nursing home labs lab diagnostics pathology pharmacy chemist medicos emergency casualty
trauma ward unit department dept wing floor gate main reception helpdesk billing counter cash canteen store
laundry mortuary theatre labour room opd ipd icu register registry log logbook pass token duty shift roster
""".split())


def has_clinical_suffix(word: str) -> bool:
    w = word.lower()
    return len(w) >= _SUFFIX_MIN_LEN and w not in _SUFFIX_EXCLUDE and bool(_CLINICAL_SUFFIX_RE.search(w))
