import sweetviz as sv
import pandas as pd

# carregamento e rotulamento dos dados
data_dir = "../"

benigns = ["output-of-benign-pcap-0.csv",
           "output-of-benign-pcap-1.csv",
           "output-of-benign-pcap-2.csv",
           "output-of-benign-pcap-3.csv"]

df_benigns = [pd.read_csv(data_dir + f, nrows=100) for f in benigns]
df_benign = pd.concat(df_benigns)
df_benign["maligno"] = 0
df_benign["tipo_maligno"] = "Benigno"

df_malware = pd.read_csv(data_dir + "output-of-malware-pcap.csv", nrows=400)
df_malware['maligno'] = 1
df_malware['tipo_maligno'] = 'Malware'

df_phishing = pd.read_csv(data_dir + "output-of-phishing-pcap.csv", nrows=400)
df_phishing['maligno'] = 1
df_phishing['tipo_maligno'] = 'Phishing'

df_spam = pd.read_csv(data_dir + "output-of-spam-pcap.csv", nrows=400)
df_spam['maligno'] = 1
df_spam['tipo_maligno'] = 'Spam'

df = pd.concat([df_benign, df_malware, df_phishing, df_spam])

# gera um relatório HTML interativo
report = sv.analyze(df)
report.show_html('relatorio_dados.html')