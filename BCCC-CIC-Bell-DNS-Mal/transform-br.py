import pandas as pd

pattern = r"\.br$"

for i in range(4):
    df_benign = pd.read_csv(f"output-of-benign-pcap-{i}.csv")
    mask = df_benign["dns_domain_name"].str.contains(pattern, regex=True)
    df_benign_br = df_benign[mask]
    df_benign_br.to_csv(f"datasets-br/output-of-benign-br-pcap-{i}.csv")
    
df_malware = pd.read_csv("output-of-malware-pcap.csv")
mask = df_malware["dns_domain_name"].str.contains(pattern, regex=True)
df_malware_br = df_malware[mask]
df_malware_br.to_csv("datasets-br/output-of-malware-br-pcap.csv")

df_phishing = pd.read_csv("output-of-phishing-pcap.csv")
mask = df_phishing["dns_domain_name"].str.contains(pattern, regex=True)
df_phishing_br = df_phishing[mask]
df_phishing_br.to_csv("datasets-br/output-of-phishing-br-pcap.csv")

df_spam = pd.read_csv("output-of-spam-pcap.csv")
mask = df_spam["dns_domain_name"].str.contains(pattern, regex=True)
df_spam_br = df_spam[mask]
df_spam_br.to_csv("datasets-br/output-of-spam-br-pcap.csv")