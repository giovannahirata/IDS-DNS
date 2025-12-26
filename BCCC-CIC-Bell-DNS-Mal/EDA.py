import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

benigns = ["output-of-benign-pcap-0.csv", 
             "output-of-benign-pcap-1.csv",
             "output-of-benign-pcap-2.csv",
             "output-of-benign-pcap-3.csv"]

df_benigns = [pd.read_csv(f) for f in benigns]
df_benign = pd.concat(df_benigns)

"""
i := input
o := output


DADOS BENIGNOS:
i: df_benign.shape
o: (3440503, 122)

>>> cols = list(df_benign.columns)
>>> cols
['Unnamed: 0', 'flow_id', 'timestamp', 'src_ip', 'src_port', 'dst_ip', 'dst_port', 'protocol', 'duration', 'packets_numbers', 'receiving_packets_numbers', 'sending_packets_numbers', 'handshake_duration', 'delta_start', 'total_bytes', 'receiving_bytes', 'sending_bytes', 'packets_rate', 'receiving_packets_rate', 'sending_packets_rate', 'packets_len_rate', 'receiving_packets_len_rate', 'sending_packets_len_rate', 'min_packets_len', 'max_packets_len', 'mean_packets_len', 'median_packets_len', 'mode_packets_len', 'standard_deviation_packets_len', 'variance_packets_len', 'coefficient_of_variation_packets_len', 'skewness_packets_len', 'min_receiving_packets_len', 'max_receiving_packets_len', 'mean_receiving_packets_len', 'median_receiving_packets_len', 'mode_receiving_packets_len', 'standard_deviation_receiving_packets_len', 'variance_receiving_packets_len', 'coefficient_of_variation_receiving_packets_len', 'skewness_receiving_packets_len', 'min_sending_packets_len', 'max_sending_packets_len', 'mean_sending_packets_len', 'median_sending_packets_len', 'mode_sending_packets_len', 'standard_deviation_sending_packets_len', 'variance_sending_packets_len', 'coefficient_of_variation_sending_packets_len', 'skewness_sending_packets_len', 'min_receiving_packets_delta_len', 'max_receiving_packets_delta_len', 'mean_receiving_packets_delta_len', 'median_receiving_packets_delta_len', 'standard_deviation_receiving_packets_delta_len', 'variance_receiving_packets_delta_len', 'mode_receiving_packets_delta_len', 'coefficient_of_variation_receiving_packets_delta_len', 'skewness_receiving_packets_delta_len', 'min_sending_packets_delta_len', 'max_sending_packets_delta_len', 'mean_sending_packets_delta_len', 'median_sending_packets_delta_len', 'standard_deviation_sending_packets_delta_len', 'variance_sending_packets_delta_len', 'mode_sending_packets_delta_len', 'coefficient_of_variation_sending_packets_delta_len', 'skewness_sending_packets_delta_len', 'max_receiving_packets_delta_time', 'mean_receiving_packets_delta_time', 'median_receiving_packets_delta_time', 'standard_deviation_receiving_packets_delta_time', 'variance_receiving_packets_delta_time', 'mode_receiving_packets_delta_time', 'coefficient_of_variation_receiving_packets_delta_time', 'skewness_sreceiving_packets_delta_time', 'min_sending_packets_delta_time', 'max_sending_packets_delta_time', 'mean_sending_packets_delta_time', 'median_sending_packets_delta_time', 'standard_deviation_sending_packets_delta_time', 'variance_sending_packets_delta_time', 'mode_sending_packets_delta_time', 'coefficient_of_variation_sending_packets_delta_time', 'skewness_sending_packets_delta_time', 'dns_domain_name', 'dns_top_level_domain', 'dns_second_level_domain', 'dns_domain_name_length', 'dns_subdomain_name_length', 'uni_gram_domain_name', 'bi_gram_domain_name', 'tri_gram_domain_name', 'numerical_percentage', 'character_distribution', 'character_entropy', 'max_continuous_numeric_len', 'max_continuous_alphabet_len', 'max_continuous_consonants_len', 'max_continuous_same_alphabet_len', 'vowels_consonant_ratio', 'conv_freq_vowels_consonants', 'distinct_ttl_values', 'ttl_values_min', 'ttl_values_max', 'ttl_values_mean', 'ttl_values_mode', 'ttl_values_variance', 'ttl_values_standard_deviation', 'ttl_values_median', 'ttl_values_skewness', 'ttl_values_coefficient_of_variation', 'distinct_A_records', 'distinct_NS_records', 'average_authority_resource_records', 'average_additional_resource_records', 'average_answer_resource_records', 'query_resource_record_type', 'ans_resource_record_type', 'query_resource_record_class', 'ans_resource_record_class', 'label']
>>> len(cols)
122


DADOS MALIGNOS:
>>> df_mal = pd.read_csv("output-of-malware-pcap.csv")
>>> df_mal.shape
(81698, 122)
>>> df_phi = pd.read_csv("output-of-phishing-pcap.csv")
>>> df_phi.shape
(43348, 122)
>>> df_spam = pd.read_csv("output-of-spam-pcap.csv")
>>> df_spam.shape
(30371, 122)

"""






df_benign0 = pd.read_csv("output-of-benign-pcap-0.csv")
df_benign1 = pd.read_csv("output-of-benign-pcap-1.csv")
df_benign2 = pd.read_csv("output-of-benign-pcap-2.csv")
df_benign3 = pd.read_csv("output-of-benign-pcap-3.csv")

columns = df_benign.columns
print(list(columns))
# print(type(columns))
print(df_benign.shape)