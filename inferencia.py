from collections import Counter
import dpkt
import math
import numpy as np
import pandas as pd
import re
import tensorflow as tf
import pickle
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # esconde avisos de info e alertas do TensorFlow

# carrega modelo e pré-processadores
model = tf.keras.models.load_model('/home/giovanna/Deteccao-de-Intrusoes-baseada-em-Perfil-Comportamental-de-DNS-utilizando-Redes-Neurais/BCCC-CIC-Bell-DNS-Mal/nn/nn1ShapPlus/bin_class/without-src_port/dns_intrusion_model_nn1.keras') # modelo de classificação binária treinado sem a src-port 
with open('/home/giovanna/Deteccao-de-Intrusoes-baseada-em-Perfil-Comportamental-de-DNS-utilizando-Redes-Neurais/BCCC-CIC-Bell-DNS-Mal/nn/nn1ShapPlus/bin_class/without-src_port/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('/home/giovanna/Deteccao-de-Intrusoes-baseada-em-Perfil-Comportamental-de-DNS-utilizando-Redes-Neurais/BCCC-CIC-Bell-DNS-Mal/nn/nn1ShapPlus/bin_class/without-src_port/selector.pkl', 'rb') as f:
    selector = pickle.load(f)
with open('/home/giovanna/Deteccao-de-Intrusoes-baseada-em-Perfil-Comportamental-de-DNS-utilizando-Redes-Neurais/BCCC-CIC-Bell-DNS-Mal/nn/nn1ShapPlus/bin_class/without-src_port/label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

def calcular_features_lexicais(dominio):
    """Calcula os metadados textuais e lexicais"""
    dominio = str(dominio).lower()
    tamanho = len(dominio)
    
    partes = dominio.split('.')
    tamanho_subdominio = 0
    tld = "not-found"
    sld = "not-found"
    
    if len(partes) > 1:
        if partes[-2] in ['com', 'org', 'net', 'edu', 'gov', 'jus', 'mil'] and len(partes) > 2:
            tld = f"{partes[-2]}.{partes[-1]}"
            sld = partes[-3]
            if len(partes) > 3:
                tamanho_subdominio = len(".".join(partes[:-3]))
        else:
            tld = partes[-1]
            sld = partes[-2]
            if len(partes) > 2:
                tamanho_subdominio = len(".".join(partes[:-2]))

    frequencias = Counter(dominio)
    entropia = -sum((count / tamanho) * math.log2(count / tamanho) for count in frequencias.values()) if tamanho > 0 else 0
    char_dist = str(dict(frequencias)) # Salva como o dicionário stringficado do treino
    
    one_gram = list(dominio)
    bi_gram = [one_gram[i] + one_gram[i+1] for i in range(len(one_gram)-1)]
    tri_gram = [one_gram[i] + one_gram[i+1] + one_gram[i+2] for i in range(len(one_gram)-2)]
    
    vogais = len(re.findall(r'[aeiou]', dominio))
    consoantes = len(re.findall(r'[bcdfghjklmnpqrstvwxyz]', dominio))
    numeros = len(re.findall(r'[0-9]', dominio))
    
    max_alpha = max([len(x) for x in re.findall(r'[a-z]+', dominio)] + [0])
    max_num = max([len(x) for x in re.findall(r'[0-9]+', dominio)] + [0])
    max_cons = max([len(x) for x in re.findall(r'[bcdfghjklmnpqrstvwxyz]+', dominio)] + [0])
    repetidas = [len(m.group(0)) for m in re.finditer(r'([a-z])\1+', dominio)]
    max_same_alpha = max(repetidas) if repetidas else 1
    
    v_c_ratio = (vogais / consoantes) if consoantes > 0 else 0.0
    num_percent = (numeros / tamanho) if tamanho > 0 else 0.0
    
    return {
        'dns_domain_name': dominio,
        'dns_top_level_domain': tld,
        'dns_second_level_domain': sld,
        'dns_domain_name_length': tamanho,
        'dns_subdomain_name_length': float(tamanho_subdominio),
        'uni_gram_domain_name': str(one_gram),
        'bi_gram_domain_name': str(bi_gram),
        'tri_gram_domain_name': str(tri_gram),
        'numerical_percentage': num_percent,
        'character_distribution': char_dist,
        'character_entropy': entropia,
        'max_continuous_numeric_len': max_num,
        'max_continuous_alphabet_len': max_alpha,
        'max_continuous_consonants_len': max_cons,
        'max_continuous_same_alphabet_len': max_same_alpha,
        'vowels_consonant_ratio': v_c_ratio
    }

def extrair_features_dns_pcap(pcap_path):
    """Lê o PCAP diretamente e extrai TTLs e Resource Records"""
    ttls, auth_rrs, add_rrs, ans_rrs = [], [], [], []
    qtypes, atypes, qclasses, aclasses = [], [], [], []
    a_records, ns_records = 0, 0
    
    try:
        with open(pcap_path, 'rb') as f:
            pcap = dpkt.pcap.Reader(f)
            for ts, buf in pcap:
                try:
                    eth = dpkt.ethernet.Ethernet(buf)
                    ip = eth.data
                    if not hasattr(ip, 'data') or type(ip.data) != dpkt.udp.UDP: continue
                    udp = ip.data
                    dns = dpkt.dns.DNS(udp.data)
                    
                    # contagem de blocos de resposta
                    ans_rrs.append(len(dns.an))
                    auth_rrs.append(len(dns.ns))
                    add_rrs.append(len(dns.ar))
                    
                    # extrai tipos de pergunta
                    for q in dns.qd:
                        qtypes.append(q.type)
                        qclasses.append(q.cls)
                        
                    # extrai dados das respostas (Answers)
                    for an in dns.an:
                        if hasattr(an, 'ttl'): ttls.append(an.ttl)
                        if hasattr(an, 'type'): 
                            atypes.append(an.type)
                            if an.type == dpkt.dns.DNS_A: a_records += 1
                            elif an.type == dpkt.dns.DNS_NS: ns_records += 1
                        if hasattr(an, 'cls'): aclasses.append(an.cls)
                        
                    # extrai TTLs das Autoridades
                    for ns in dns.ns:
                        if hasattr(ns, 'ttl'): ttls.append(ns.ttl)
                except:
                    continue
    except Exception as e:
        print(f"Aviso ao ler PCAP: {e}")

    # Cálculos estatísticos
    s_ttls = pd.Series(ttls) if len(ttls) > 0 else pd.Series([0.0])
    
    mean_val = float(s_ttls.mean())
    std_val = float(s_ttls.std(ddof=0))
    var_val = float(s_ttls.var(ddof=0))
    skew_val = float(s_ttls.skew()) if var_val > 0 else 0.0
    cv_val = (std_val / mean_val) if mean_val > 0 else 0.0
    
    return {
        'distinct_ttl_values': len(s_ttls.unique()),
        'ttl_values_min': float(s_ttls.min()),
        'ttl_values_max': float(s_ttls.max()),
        'ttl_values_mean': mean_val,
        'ttl_values_mode': float(s_ttls.mode()[0]) if not s_ttls.mode().empty else 0.0,
        'ttl_values_variance': var_val,
        'ttl_values_standard_deviation': std_val,
        'ttl_values_median': float(s_ttls.median()),
        'ttl_values_skewness': skew_val,
        'ttl_values_coefficient_of_variation': cv_val,
        'distinct_A_records': a_records,
        'distinct_NS_records': ns_records,
        'average_authority_resource_records': float(np.mean(auth_rrs)) if auth_rrs else 0.0,
        'average_additional_resource_records': float(np.mean(add_rrs)) if add_rrs else 0.0,
        'average_answer_resource_records': float(np.mean(ans_rrs)) if ans_rrs else 0.0,
        'query_resource_record_type': str(qtypes),
        'ans_resource_record_type': str(atypes),
        'query_resource_record_class': str(qclasses),
        'ans_resource_record_class': str(aclasses)
    }

def preprocess_new_data(df_new, dominio_alvo, csv_path, pcap_path):
    """Pré-processa dados para inferencia/predição injetando métricas lexicais"""
    
    colunas_treinamento = [
        'flow_id', 'timestamp', 'src_ip', 'src_port', 'dst_ip', 'dst_port', 'protocol', 'duration', 
        'packets_numbers', 'receiving_packets_numbers', 'sending_packets_numbers', 'handshake_duration', 
        'delta_start', 'total_bytes', 'receiving_bytes', 'sending_bytes', 'packets_rate', 
        'receiving_packets_rate', 'sending_packets_rate', 'packets_len_rate', 'receiving_packets_len_rate', 
        'sending_packets_len_rate', 'min_packets_len', 'max_packets_len', 'mean_packets_len', 
        'median_packets_len', 'mode_packets_len', 'standard_deviation_packets_len', 'variance_packets_len', 
        'coefficient_of_variation_packets_len', 'skewness_packets_len', 'min_receiving_packets_len', 
        'max_receiving_packets_len', 'mean_receiving_packets_len', 'median_receiving_packets_len', 
        'mode_receiving_packets_len', 'standard_deviation_receiving_packets_len', 'variance_receiving_packets_len', 
        'coefficient_of_variation_receiving_packets_len', 'skewness_receiving_packets_len', 
        'min_sending_packets_len', 'max_sending_packets_len', 'mean_sending_packets_len', 
        'median_sending_packets_len', 'mode_sending_packets_len', 'standard_deviation_sending_packets_len', 
        'variance_sending_packets_len', 'coefficient_of_variation_sending_packets_len', 'skewness_sending_packets_len', 
        'min_receiving_packets_delta_len', 'max_receiving_packets_delta_len', 'mean_receiving_packets_delta_len', 
        'median_receiving_packets_delta_len', 'standard_deviation_receiving_packets_delta_len', 
        'variance_receiving_packets_delta_len', 'mode_receiving_packets_delta_len', 
        'coefficient_of_variation_receiving_packets_delta_len', 'skewness_receiving_packets_delta_len', 
        'min_sending_packets_delta_len', 'max_sending_packets_delta_len', 'mean_sending_packets_delta_len', 
        'median_sending_packets_delta_len', 'standard_deviation_sending_packets_delta_len', 
        'variance_sending_packets_delta_len', 'mode_sending_packets_delta_len', 
        'coefficient_of_variation_sending_packets_delta_len', 'skewness_sending_packets_delta_len', 
        'max_receiving_packets_delta_time', 'mean_receiving_packets_delta_time', 'median_receiving_packets_delta_time', 
        'standard_deviation_receiving_packets_delta_time', 'variance_receiving_packets_delta_time', 
        'mode_receiving_packets_delta_time', 'coefficient_of_variation_receiving_packets_delta_time', 
        'skewness_sreceiving_packets_delta_time', 'min_sending_packets_delta_time', 'max_sending_packets_delta_time', 
        'mean_sending_packets_delta_time', 'median_sending_packets_delta_time', 'standard_deviation_sending_packets_delta_time', 
        'variance_sending_packets_delta_time', 'mode_sending_packets_delta_time', 
        'coefficient_of_variation_sending_packets_delta_time', 'skewness_sending_packets_delta_time', 
        'dns_domain_name', 'dns_top_level_domain', 'dns_second_level_domain', 'dns_domain_name_length', 
        'dns_subdomain_name_length', 'uni_gram_domain_name', 'bi_gram_domain_name', 'tri_gram_domain_name', 
        'numerical_percentage', 'character_distribution', 'character_entropy', 'max_continuous_numeric_len', 
        'max_continuous_alphabet_len', 'max_continuous_consonants_len', 'max_continuous_same_alphabet_len', 
        'vowels_consonant_ratio', 'conv_freq_vowels_consonants', 'distinct_ttl_values', 'ttl_values_min', 
        'ttl_values_max', 'ttl_values_mean', 'ttl_values_mode', 'ttl_values_variance', 'ttl_values_standard_deviation', 
        'ttl_values_median', 'ttl_values_skewness', 'ttl_values_coefficient_of_variation', 'distinct_A_records', 
        'distinct_NS_records', 'average_authority_resource_records', 'average_additional_resource_records', 
        'average_answer_resource_records', 'query_resource_record_type', 'ans_resource_record_type', 
        'query_resource_record_class', 'ans_resource_record_class'
    ]

    # Tratamento numérico inicial 
    colunas_sujas = ['handshake_duration', 'delta_start']
    for col in colunas_sujas:
        if col in df_new.columns:
            df_new[col] = pd.to_numeric(df_new[col], errors='coerce').fillna(0.0)
            
    # Injeta as features lexicais
    features_calculadas = calcular_features_lexicais(dominio_alvo)
    for feature, valor in features_calculadas.items():
        df_new[feature] = valor

    # Injeta as features ttl
    features_dns = extrair_features_dns_pcap(pcap_path)
    for feature, valor in features_dns.items():
        df_new[feature] = valor

    # Cria colunas faltantes com 0 e reordena exatamente como no treino
    for col in colunas_treinamento:
        if col not in df_new.columns:
            df_new[col] = 0.0
            
    df_new = df_new[colunas_treinamento].copy().fillna(0)
    
    # salva csv na forma natural dos dados brutos
    caminho_natural = csv_path.replace('.csv', '_NATURAL.csv')
    df_new.to_csv(caminho_natural, index=False)
    print(f"[*] CSV natural (com texto) salvo para auditoria em: {caminho_natural}")

    # encoding categorico
    # ignora o dtype e itera diretamente pelo que o LabelEncoder conhece
    if isinstance(label_encoders, dict):
        for col, le in label_encoders.items():
            if col in df_new.columns:
                df_new[col] = df_new[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )

    # depurador de encoding
    print(f"\n[DEBUG] ID do domínio codificado: {df_new['dns_domain_name'].iloc[0]}")
    print(f"[DEBUG] ID dos uni-grams codificados: {df_new['uni_gram_domain_name'].iloc[0]}")
    
    # Limpeza (transforma protocolos como "DNS" em 0.0 se não foram encodados)
    for col in df_new.columns:
        if df_new[col].dtype == 'object' or df_new[col].dtype.name == 'string':
            df_new[col] = pd.to_numeric(df_new[col], errors='coerce').fillna(0.0)

    # Salva CSV de auditoria
    caminho_auditoria = csv_path.replace('.csv', '_LIMPO.csv')
    df_new.to_csv(caminho_auditoria, index=False)
    print(f"[*] CSV limpo e corrigido salvo para auditoria em: {caminho_auditoria}")

    # Normalização (Scaler)
    numerical_cols_esperadas = scaler.feature_names_in_
    for col in numerical_cols_esperadas:
        df_new[col] = pd.to_numeric(df_new[col], errors='coerce').fillna(0.0)
        
    df_new[numerical_cols_esperadas] = scaler.transform(df_new[numerical_cols_esperadas])
    
    # Seleção de features (Selector)
    features_esperadas = selector.feature_names_in_
    for col in features_esperadas:
        df_new[col] = pd.to_numeric(df_new[col], errors='coerce').fillna(0.0)
        
    x_new = selector.transform(df_new[features_esperadas])
    
    return np.array(x_new, dtype=np.float32)

def predict(df_new, dominio_alvo, csv_path, pcap_path):
    """Faz inferencias/predições nos dados"""
    x_processed = preprocess_new_data(df_new, dominio_alvo, csv_path, pcap_path)
    predictions = model.predict(x_processed)
    return (predictions > 0.5).astype(int).ravel()

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Uso: python3 inferencia.py <caminho_csv> <dominio> <caminho_pcap>")
        sys.exit(1)
        
    csv_path = sys.argv[1]
    dominio_alvo = sys.argv[2]
    pcap_path = sys.argv[3]
    
    print(f"[*] CSV carregado para inferência: {csv_path}")
    df_test = pd.read_csv(csv_path)
    
    predictions = predict(df_test, dominio_alvo, csv_path, pcap_path)
    
    print("Predições:")
    print(predictions)
    print(f"Benigno: {np.sum(predictions == 0)}")
    print(f"Maligno: {np.sum(predictions == 1)}")