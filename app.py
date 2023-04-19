from flask import Flask, render_template, request, redirect, jsonify, url_for
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress tensorflow warnings
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from keras import models
import numpy as np
from PIL import Image


app = Flask(__name__)

CLASSES = ['accipiter_gentilis','accipiter_nisus','acrocephalus_arundinaceus','acrocephalus_melanopogon','acrocephalus_palustris','acrocephalus_schoenobaenus','acrocephalus_scirpaceus','actitis_hypoleucos','aegithalos_caudatus','aegolius_funereus','aegypius_monachus','alauda_arvensis','alcedo_atthis','alectoris_barbara','alectoris_graeca','alectoris_rufa','anas_acuta','anas_clypeata','anas_crecca','anas_discors','anas_formosa','anas_penelope','anas_platyrhynchos','anas_querquedula','anas_strepera','anser_albifrons','anser_anser','anser_erythropus','anser_fabalis','anthus_campestris','anthus_cervinus','anthus_pratensis','anthus_spinoletta','anthus_trivialis','apus_apus','apus_pallidus','aquila_chrysaetos','aquila_clanga','aquila_pomarina','ardea_alba','ardea_cinerea','ardea_purpurea','ardeola_ralloides','arenaria_interpres','asio_flammeus','asio_otus','athene_noctua','aythya_ferina','aythya_fuligula','aythya_marila','aythya_nyroca','bombycilla_garrulus','botaurus_stellaris','branta_bernicla','branta_leucopsis','branta_ruficollis','bubo_bubo','bubulcus_ibis','bucanetes_githagineus','bucephala_clangula','burhinus_oedicnemus','buteo_buteo','buteo_lagopus','buteo_rufinus','calandrella_brachydactyla','calcarius_lapponicus','calidris_alba','calidris_alpina','calidris_canutus','calidris_ferruginea','calidris_minuta','calidris_temminckii','calonectris_diomedea','caprimulgus_europaeus','carduelis_cannabina','carduelis_carduelis','carduelis_chloris','carduelis_citrinella','carduelis_corsicana','carduelis_flammea','carduelis_spinus','carpodacus_erythrinus','certhia_brachydactyla','certhia_familiaris','cettia_cetti','charadrius_alexandrinus','charadrius_dubius','charadrius_hiaticula','charadrius_morinellus','chlidonias_hybridus','chlidonias_leucopterus','chlidonias_niger','chroicocephalus_genei','chroicocephalus_ridibundus','ciconia_ciconia','ciconia_nigra','cinclus_cinclus','circaetus_gallicus','circus_aeruginosus','circus_cyaneus','circus_macrourus','circus_pygargus','cisticola_juncidis','clamator_glandarius','clangula_hyemalis','coccothraustes_coccothraustes','coloeus_monedula','columba_livia','columba_oenas','columba_palumbus','coracias_garrulus','corvus_corax','corvus_corone','corvus_frugilegus','coturnix_coturnix','crex_crex','cuculus_canorus','cursorius_cursor','cygnus_bewickii','cygnus_cygnus','cygnus_olor','delichon_urbicum','dendrocopos_leucotos','dendrocopos_major','dendrocopos_medius','dendrocopos_minor','dryocopus_martius','egretta_garzetta','emberiza_cia','emberiza_cirlus','emberiza_citrinella','emberiza_hortulana','emberiza_leucocephalos','emberiza_melanocephala','emberiza_pusilla','emberiza_schoeniclus','eremophila_alpestris','erithacus_rubecula','falco_biarmicus','falco_cherrug','falco_columbarius','falco_eleonorae','falco_naumanni','falco_peregrinus','falco_subbuteo','falco_tinnunculus','falco_vespertinus','ficedula_albicollis','ficedula_hypoleuca','ficedula_parva','ficedula_semitorquata','francolinus_francolinus','fringilla_coelebs','fringilla_montifringilla','fulica_atra','galerida_cristata','gallinago_gallinago','gallinago_media','gallinula_chloropus','garrulus_glandarius','gavia_arctica','gavia_immer','gavia_stellata','gelochelidon_nilotica','glareola_pratincola','glaucidium_passerinum','grus_grus','gypaetus_barbatus','gyps_fulvus','haematopus_ostralegus','haliaeetus_albicilla','hieraaetus_fasciatus','hieraaetus_pennatus','himantopus_himantopus','hippolais_icterina','hippolais_polyglotta','hirundo_daurica','hirundo_rustica','histrionicus_histrionicus','ichthyaetus_audouinii','ichthyaetus_melanocephalus','ixobrychus_minutus','jynx_torquilla','lagopus_mutus','lanius_collurio','lanius_excubitor','lanius_minor','lanius_phoenicuroides','lanius_senator','larus_argentatus','larus_cachinnans','larus_canus','larus_fuscus','larus_hyperboreus','larus_ichthyaetus','larus_marinus','larus_michahellis','larus_minutus','limicola_falcinellus','limosa_lapponica','limosa_limosa','locustella_luscinioides','locustella_naevia','loxia_curvirostra','lullula_arborea','luscinia_luscinia','luscinia_megarhynchos','luscinia_svecica','lymnocryptes_minimus','macronectes_giganteus','marmaronetta_angustirostris','melanitta_fusca','melanitta_nigra','melanocorypha_calandra','mergellus_albellus','mergus_merganser','mergus_serrator','merops_apiaster','microcarbo_pygmeus','miliaria_calandra','milvus_migrans','milvus_milvus','monticola_saxatilis','monticola_solitarius','montifringilla_nivalis','morus_bassanus','motacilla_alba','motacilla_cinerea','motacilla_flava','muscicapa_striata','neophron_percnopterus','netta_rufina','nucifraga_caryocatactes','numenius_arquata','numenius_phaeopus','nycticorax_nycticorax','oenanthe_hispanica','oenanthe_isabellina','oenanthe_oenanthe','oriolus_oriolus','otis_tarda','otus_scops','oxyura_jamaicensis','oxyura_leucocephala','pandion_haliaetus','panurus_biarmicus','parus_ater','parus_caeruleus','parus_cristatus','parus_major','parus_montanus','parus_palustris','passer_domesticus','passer_hispaniolensis','passer_italiae','passer_montanus','pelecanus_onocrotalus','perdix_perdix','pernis_apivorus','petronia_petronia','phalacrocorax_aristotelis','phalacrocorax_carbo','phalaropus_lobatus','phasianus_colchicus','philomachus_pugnax','phoenicopterus_roseus','phoenicurus_ochruros','phoenicurus_phoenicurus','phylloscopus_bonelli','phylloscopus_collybita','phylloscopus_inornatus','phylloscopus_proregulus','phylloscopus_sibilatrix','phylloscopus_trochilus','pica_pica','picoides_tridactylus','picus_canus','picus_viridis','platalea_leucorodia','plectrophenax_nivalis','plegadis_falcinellus','pluvialis_apricaria','pluvialis_squatarola','podiceps_auritus','podiceps_cristatus','podiceps_grisegena','podiceps_nigricollis','porphyrio_porphyrio','porzana_parva','porzana_porzana','prunella_collaris','prunella_modularis','ptyonoprogne_rupestris','puffinus_yelkouan','pyrrhocorax_graculus','pyrrhocorax_pyrrhocorax','pyrrhula_pyrrhula','rallus_aquaticus','recurvirostra_avosetta','regulus_ignicapillus','regulus_regulus','remiz_pendulinus','riparia_riparia','rissa_tridactyla','saxicola_rubetra','saxicola_torquata','scolopax_rusticola','serinus_serinus','sinosuthora_webbiana','sitta_europaea','somateria_mollissima','somateria_spectabilis','stercorarius_longicaudus','stercorarius_parasiticus','stercorarius_pomarinus','stercorarius_skua','sterna_caspia','sterna_hirundo','sternula_albifrons','streptopelia_decaocto','streptopelia_turtur','strix_aluco','strix_uralensis','sturnus_roseus','sturnus_unicolor','sturnus_vulgaris','sula_leucogaster','sylvia_atricapilla','sylvia_borin','sylvia_cantillans','sylvia_communis','sylvia_conspicillata','sylvia_curruca','sylvia_hortensis','sylvia_melanocephala','sylvia_nisoria','sylvia_rueppelli','sylvia_sarda','sylvia_undata','tachybaptus_ruficollis','tachymarptis_melba','tadorna_ferruginea','tadorna_tadorna','tetrao_tetrix','tetrao_urogallus','tetrastes_bonasia','tetrax_tetrax','thalasseus_sandvicensis','threskiornis_aethiopica','tichodroma_muraria','tringa_erythropus','tringa_glareola','tringa_nebularia','tringa_ochropus','tringa_stagnatilis','tringa_totanus','troglodytes_troglodytes','turdus_iliacus','turdus_merula','turdus_philomelos','turdus_pilaris','turdus_torquatus','turdus_viscivorus','tyto_alba','upupa_epops','vanellus_gregarius','vanellus_vanellus','xenus_cinereus']
CLASSES.sort()

first_run = True

# Initialize the predictor model
model = None

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    global first_run

    if first_run:
        load_classifier()
        first_run = False

    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_image():
    img = request.files['image']
    img_preprocessed = image_preprocessing(img)
    predicted_species = compute_exact_prediction(img_preprocessed)
    compute_probability_distribution(img_preprocessed)

    if predicted_species == None:
        return redirect(url_for('result', species="not_able_to_predict"))
    return redirect(url_for('result', species=predicted_species))

@app.route('/result')
def result():
    species = request.args.get('species')

    if species == "not_able_to_predict":
        return render_template('error.html')

    species = species.replace("_"," ")
    species = species[0].upper() + species[1:]
    print("\nOne shot guess: the predicted species is " + species)

    return render_template('result.html', species=species)

def load_classifier():
    global model
    model = models.load_model('./models/Fine_Tuning_New_Epochs_765432_80-05_1-63.h5', compile=False)


def resize_image(image_path, width, height):
    with Image.open(image_path) as image:
        # get the original image size
        original_width, original_height = image.size

        # calculate the aspect ratios
        width_ratio = 224 / original_width
        height_ratio = 224 / original_height

        # find the smallest ratio to resize the image without stretching or squeezing it
        resize_ratio = min(width_ratio, height_ratio)

        # calculate the new image size
        new_width = int(original_width * resize_ratio)
        new_height = int(original_height * resize_ratio)

        # resize the image
        resized_image = image.resize((new_width, new_height), Image.ANTIALIAS)

        # create a new blank image of the desired shape
        final_image = Image.new('RGB', (224, 224))

        # calculate the crop coordinates
        left = (224 - new_width) // 2
        top = (224 - new_height) // 2
        right = left + new_width
        bottom = top + new_height

        # paste the resized image into the blank image
        final_image.paste(resized_image, (left, top))

        return final_image


def image_preprocessing(image_path):
    img = resize_image(image_path, 224, 224)
    # img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)
    return img_preprocessed

def compute_exact_prediction(img_preprocessed):
    global CLASSES, model

    # One-shot prediction:
    try:
        index_label = int(np.argmax(model.predict(img_preprocessed), axis=-1))
    except:
        print("An error occurred while resizing the submitted image")
        return

    label_name = CLASSES[index_label]

    return label_name

def compute_probability_distribution(img_preprocessed):
    global model

    prediction = model.predict(img_preprocessed)
    rates = np.fliplr(np.sort(prediction, axis=1)[:, -3:])
    categories = np.fliplr(np.argsort(prediction, axis=1)[:, -3:])
    rates = list(rates[0])
    categories = list(categories[0])
    for index_label, prob in zip(categories, rates):
        print("La specie appartiene a " + str(CLASSES[index_label]) + " con probabilità " + "{:.9%}".format(prob))

    return


if __name__ == '__main__':

    load_classifier()

    #selected_image_path = "./testing-images/aegithalos_caudatus.jpg"

    #img_preprocessed = image_preprocessing(selected_image_path)

    # One-shot prediction:
    #compute_exact_prediction(img_preprocessed)

    # Prob Distribution of prediction
    #compute_probability_distribution(img_preprocessed)

    app.run(debug=False, host="0.0.0.0")
