import pandas as pd

dir = "~/Deteccao-de-Intrusoes-baseada-em-Perfil-Comportamental-de-DNS-utilizando-Redes-Neurais/BCCC-CIC-Bell-DNS-Mal/datasets-br/"

benignos = ["output-of-benign-br-pcap-0.csv",
            "output-of-benign-br-pcap-1.csv",
            "output-of-benign-br-pcap-2.csv",
            "output-of-benign-br-pcap-3.csv"]
df_benignos = [pd.read_csv(dir+f) for f in benignos]
df_benigno = pd.concat(df_benignos)
dominios_benignos = df_benigno["dns_domain_name"].unique()

try:
    with open("lista-dominios-br.txt", "x") as f:
        f.write("Domínios benignos:\n")
        f.write("\n".join(f'"{dominio}"' for dominio in dominios_benignos))
        f.write("\n\n\n")
except FileExistsError:
    with open("lista-dominios-br.txt", "a") as f:
        f.write("Domínios benignos:\n")
        f.write("\n".join(f'"{dominio}"' for dominio in dominios_benignos))
        f.write("\n\n\n")

df_malware = pd.read_csv(dir+"output-of-malware-br-pcap.csv")
dominios_malware = df_malware["dns_domain_name"].unique()

with open("lista-dominios-br.txt", "a") as f:
    f.write("Domínios malware:\n")
    f.write("\n".join(f'"{dominio}"' for dominio in dominios_malware))
    f.write("\n\n\n")

df_phishing = pd.read_csv(dir+"output-of-phishing-br-pcap.csv")
dominios_phishing = df_phishing["dns_domain_name"].unique()

with open("lista-dominios-br.txt", "a") as f:
    f.write("Domínios phishing:\n")
    f.write("\n".join(f'"{dominio}"' for dominio in dominios_phishing))
    f.write("\n\n\n")

df_spam = pd.read_csv(dir+"output-of-spam-br-pcap.csv")
dominios_spam = df_spam["dns_domain_name"].unique()

with open("lista-dominios-br.txt", "a") as f:
    f.write("Domínios spam:\n")
    f.write("\n".join(f'"{dominio}"' for dominio in dominios_spam))
    f.write("\n\n\n")
