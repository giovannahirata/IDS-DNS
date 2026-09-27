import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

# ============================================================================
# CARREGAMENTO DOS ARTEFATOS DO MELHOR MODELO EM "melhor-geral/"
# ============================================================================

def load_artifacts_geral(base_dir="melhor-geral"):
    """Carrega explicitamente os artefatos da pasta melhor-geral."""
    model_path = os.path.join(base_dir, 'model_10.keras')
    scaler_path = os.path.join(base_dir, 'scaler.pkl')
    selector_path = os.path.join(base_dir, 'selector.pkl')
    le_path = os.path.join(base_dir, 'label_encoders.pkl')

    for path in [model_path, scaler_path, selector_path, le_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Arquivo obrigatório não encontrado: {path}")

    print(f"Carregando modelo: {model_path}")
    model = tf.keras.models.load_model(model_path)

    print("Carregando pré-processadores (.pkl)...")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(selector_path, 'rb') as f:
        selector = pickle.load(f)
    with open(le_path, 'rb') as f:
        label_encoders = pickle.load(f)

    return model, scaler, selector, label_encoders


def safe_label_transform(le, series):
    """Trata categorias não vistas no treino usando a primeira classe como fallback."""
    known_classes = set(le.classes_)
    fallback_value = le.classes_[0]
    series_cleaned = series.astype(str).map(lambda x: x if x in known_classes else fallback_value)
    return le.transform(series_cleaned)


def preprocess_data(df_raw, label_encoders, scaler, selector, target_cols=['classe_real', 'tipo_real']):
    """
    Executa o pipeline original mantendo dns_domain_name, pois ele é esperado
    pelo label_encoder e pelo scaler.
    """
    df_clean = df_raw.copy()
    
    # Remove apenas metadados de tráfego que nunca fizeram parte do modelo
    cols_remove = ['flow_id', 'timestamp', 'src_ip', 'dst_ip', 'src_port', 'dst_port', 'label', 'maligno', 'tipo_maligno']
    cols_to_drop = [c for c in df_clean.columns if c in cols_remove or c.startswith('Unnamed') or c in target_cols]
    x = df_clean.drop(columns=cols_to_drop, errors='ignore')
    
    # Trata valores nulos
    x = x.fillna(0)

    # Label Encoding (inclui dns_domain_name)
    categorical_cols = [c for c in x.select_dtypes(include=['object', 'string']).columns]
    for col in categorical_cols:
        if col in label_encoders:
            x[col] = safe_label_transform(label_encoders[col], x[col])

    # Normalização com o scaler
    num_cols = x.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns
    x[num_cols] = scaler.transform(x[num_cols])

    # Seleção de atributos
    x_transformed = selector.transform(x)
    return np.array(x_transformed, dtype=np.float32)


# ============================================================================
# LISTA DE DOMÍNIOS
# ============================================================================

dominios_alvo = [
    "acesseweb.com.br", "aromadecafe.com.br", "supermercadosbandeirante.com.br", 
    "bolsavirtual.com.br", "pontoapontocondominios.com.br", "brasilcom.com.br", 
    "vixface.com.br", "catanduvashow.com.br", "midianews.com.br", "jmedhc.com.br", 
    "laboratoriovision.com.br", "hostdados.com.br", "mondialeacrilicos.com.br", 
    "bilheterama.com.br", "ferramentasindustriais.com.br", "fraudespam.blogspot.com.br", 
    "rolamentosrs.com.br", "creativeonline.com.br", "construmaxservicos.com.br", 
    "dietec.com.br", "neintercambio.com.br", "br37.dialhost.com.br", 
    "aulasdeinglesporskype.com.br", "onedigital.com.br", "cidvale.com.br", 
    "envionews.com.br", "madeireirafarias.com.br", "tekagrafica.com.br", 
    "brsafe.com.br", "dessa.com.br", "disney.com.br", "jocecabeleireiros.com.br", 
    "softdownload.com.br", "minabella.com.br", "migalhas.com.br", "valoronline.com.br", 
    "seguidores.com.br", "krambola.com.br", "quadrus.com.br", "noticiasecuriosidades.com.br", 
    "santavita.com.br", "grupoct.com.br", "magazinevoce.com.br", "ouvirmusica.com.br", 
    "2018.cbv.com.br", "minhasinscricoes.com.br", "cervejariacacique.com.br", 
    "cvssistemas.com.br", "oftalmologiahigienopolis.com.br", "uploaddeimagens.com.br", 
    "webnode.com.br", "superbandeirante.com.br", "dric-arcondicionado.com.br", 
    "falcaobatidos.com.br", "reidascalhas.net.br", "tam.com.br", "itforum365.com.br", 
    "agenciacatraca.com.br", "imobiliariauniao.com.br", "locaweb.com.br", 
    "webmail-seguro.com.br", "santuariopalacehotel.com.br", "verdesmares.com.br", 
    "pousadacasadoangelo.com.br", "meioemensagem.com.br", "sirrus.com.br", 
    "cadastro.supermercadocolatusso.com.br", "whatsdoor.com.br", "connectypay.com.br", 
    "oreidoimportado.com.br", "chiesautomoveis.com.br", "www.britoveiculos.com.br", 
    "corretoraexecutiva.com.br", "blogvejaagora.com.br", "sanchesblanes.com.br", 
    "www.tari.com.br", "consultoriaemestetica.com.br", "cbcengenharia.com.br", 
    "munizadvocacia.adv.br", "alistanegra.com.br", "fucapet.com.br", "afilio.com.br", 
    "beckerseguros.com.br", "legisweb.com.br", "coop.br", "esp.br", "otimaideia.com.br", 
    "em.com.br", "www.vidrosmichaelsen.com.br", "kinghost.com.br", 
    "bandeirantessupermercado.com.br", "superdownloads.com.br", "palcomp3.com.br", 
    "mail.mercadodocacau.com.br", "sacoeumsaco.com.br", "omestredahq.com.br", 
    "itau-unibanco.com.br", "safetylifeblindagens.com.br", "faccondominios.com.br", 
    "dsladvogados.com.br", "zipmail.uol.com.br", "sony.com.br", "rioverdepar.com.br", 
    "conjur.com.br", "o2speed.com.br", "kaebischschokoladen.com.br", "orkut.com.br", 
    "jornalgazeta.com.br", "manutencaopreventiva.com.br", "vagas.com.br", 
    "invistaconstrutora.com.br", "2mnoticias.com.br", "poderdafala.com.br", 
    "ubsistemas.com.br", "siteapp.com.br", "justicaemacao.com.br", "intcursos.com.br", 
    "bbrsolucoes.com.br", "confaeb.com.br", "belacruz.ce.gov.br", "algotextil.com.br", 
    "hotelterraviva.com.br", "alecrimatelie.com.br", "mvhp.com.br", "polimentosroberto.com.br", 
    "geralinks.com.br", "k8.com.br", "defaconstrutora.com.br", "guarusite.com.br", 
    "bslsaude.com.br", "conectalitoral.com.br", "acate.com.br", "ecofinders.com.br", 
    "masterempresas.com.br", "www.mercadobitcoin.com.br", "camaradeaugustodelima.mg.gov.br", 
    "adconminas.com.br", "ingressorapido.com.br", "ffdevweb.com.br", "formatoib.com.br", 
    "pousadamarambaia.com.br", "multimport-rs.com.br", "recuperacaojudicialuberaba.com.br", 
    "peteleco.com.br", "centralinternet.com.br", "univercidades.org.br", "maekawa.adv.br", 
    "resultadosdigitais.com.br", "netplaca.com.br", "cafedonasantina.com.br", 
    "diskterra.com.br", "sevenstreet.com.br", "embreara.com.br", "eventbrite.com.br", 
    "nandabolsas.com.br", "culturabancodobrasil.com.br", "javalipecas.com.br", 
    "oxautomacao.com.br", "trindadepecas.com.br", "estanciadapicanha.com.br", 
    "periautomoveis.com.br", "bioline.org.br", "olhardigital.com.br", "clickgratis.com.br", 
    "monetizze.com.br", "mknetwork.com.br", "www.manutencaodecompressores.com.br", 
    "uberveiculos.com.br", "icomaq.com.br", "pesquisesuaviagem.com.br", "dintec.com.br", 
    "agenciaw3.com.br", "santavitta.com.br", "ndbrinquedos.com.br", "quebarato.com.br", 
    "www.plastpell.com.br", "captativa.com.br", "itnowitau.com.br", "iddeia.org.br", 
    "renovaleplanejados.com.br", "ignifire.com.br", "segredosdaaudiencia.com.br", 
    "paar.com.br", "astreinbrasil.com.br", "pastazip.com.br", "telhasbetel.com.br", 
    "www.fulbra.org.br", "colegioanchieta.org.br", "www.casamiracolli.com.br", 
    "cadastrointernet.com.br", "4sigmas.com.br", "bolsapromocional.com.br", 
    "ajudadireito.com.br", "tessarolomarmores.com.br", "poraodigital.com.br", 
    "youse.com.br", "pressplay.com.br", "bucli.com.br", "kiamatriz.com.br", 
    "manutencaodecompressores.com.br"
]

# ============================================================================
# INFERÊNCIA
# ============================================================================

model, scaler, selector, label_encoders = load_artifacts_geral("melhor-geral")

dataset_dir = "/home/giovanna/Deteccao-de-Intrusoes-baseada-em-Perfil-Comportamental-de-DNS-utilizando-Redes-Neurais/BCCC-CIC-Bell-DNS-Mal/datasets-br/"
test_files = [
    dataset_dir + "output-of-benign-br-pcap-0.csv",
    dataset_dir + "output-of-benign-br-pcap-1.csv",
    dataset_dir + "output-of-benign-br-pcap-2.csv",
    dataset_dir + "output-of-benign-br-pcap-3.csv",
    dataset_dir + "output-of-malware-br-pcap.csv",
    dataset_dir + "output-of-phishing-br-pcap.csv",
    dataset_dir + "output-of-spam-br-pcap.csv"
]

print("Lendo datasets e filtrando apenas as linhas necessárias...")
df_list = []
for f in test_files:
    if os.path.exists(f):
        df_temp = pd.read_csv(f)
        
        df_matched = df_temp[df_temp['dns_domain_name'].isin(dominios_alvo)].copy()
        
        if not df_matched.empty:
            fname = f.lower()
            if 'benign' in fname:
                df_matched['classe_real'] = 'Benigno'
                df_matched['tipo_real'] = 'Benigno'
            elif 'malware' in fname:
                df_matched['classe_real'] = 'Maligno'
                df_matched['tipo_real'] = 'Malware'
            elif 'phishing' in fname:
                df_matched['classe_real'] = 'Maligno'
                df_matched['tipo_real'] = 'Phishing'
            elif 'spam' in fname:
                df_matched['classe_real'] = 'Maligno'
                df_matched['tipo_real'] = 'Spam'
            else:
                df_matched['classe_real'] = 'Desconhecido'
                df_matched['tipo_real'] = 'Desconhecido'
                
            df_list.append(df_matched)

if not df_list:
    raise ValueError("Nenhum domínio da lista foi encontrado nos arquivos CSV.")

df_test_filtered = pd.concat(df_list, ignore_index=True)

# Armazena metadados para o relatório final
nomes_dominios = df_test_filtered['dns_domain_name'].values
classes_reais = df_test_filtered['classe_real'].values
tipos_reais = df_test_filtered['tipo_real'].values

# Pré-processamento sem perder features
X_proc = preprocess_data(df_test_filtered, label_encoders, scaler, selector)

# Predição no Keras
probs = model.predict(X_proc, verbose=0).ravel()

# Montagem do DataFrame comparativo
df_resultados = pd.DataFrame({
    'dns_domain_name': nomes_dominios,
    'classe_real': classes_reais,
    'tipo_real': tipos_reais,
    'prob_maligno': probs,
    'predicao_modelo': np.where(probs > 0.5, 'Maligno', 'Benigno')
})

# Consolidação (caso o mesmo domínio apareça em múltiplos fluxos de tráfego)
df_final = df_resultados.groupby('dns_domain_name').agg({
    'classe_real': 'first',
    'tipo_real': 'first',
    'prob_maligno': 'mean',
    'predicao_modelo': lambda x: 'Maligno' if (x == 'Maligno').mean() >= 0.5 else 'Benigno'
}).reset_index()

df_final['resultado'] = np.where(df_final['classe_real'] == df_final['predicao_modelo'], 'Correto', 'Erro')
df_final['prob_maligno'] = df_final['prob_maligno'].round(4)

print("\n" + "="*85)
print("INFERÊNCIA COM model_10.keras (melhor-geral)")
print("="*85)
print(df_final.to_string(index=False))

output_csv = "resultado_model_10_geral.csv"
df_final.to_csv(output_csv, index=False)
print(f"\nArquivo salvo com sucesso em: '{output_csv}'")