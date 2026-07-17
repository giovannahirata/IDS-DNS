import dpkt
import sys

# recebe os parâmetros enviados pelo script bash 
if len(sys.argv) < 4:
    print("Uso: python3 limpa_pcap.py <pcap_entrada> <pcap_saida> <dominio>")
    sys.exit(1)

pcap_input = sys.argv[1]
pcap_output = sys.argv[2]
dominio_alvo = sys.argv[3]

def obter_leitor(f):
    try:
        f.seek(0)
        leitor = dpkt.pcapng.Reader(f)
        _ = next(iter(leitor))
        f.seek(0)
        return dpkt.pcapng.Reader(f)
    except Exception:
        f.seek(0)
        return dpkt.pcap.Reader(f)

# Descobre os IDs legítimos do domínio
ids_legitimos = set()
with open(pcap_input, 'rb') as f:
    pcap = obter_leitor(f)
    for ts, buf in pcap:
        try:
            eth = dpkt.ethernet.Ethernet(buf)
            ip = eth.data
            udp = ip.data
            dns = dpkt.dns.DNS(udp.data)
            
            for q in dns.qd:
                if dominio_alvo in q.name:
                    ids_legitimos.add(dns.id)
        except:
            continue

# Salva o PCAP casando Pergunta e Resposta
with open(pcap_input, 'rb') as f_in, open(pcap_output, 'wb') as f_out:
    pcap_in = obter_leitor(f_in)
    pcap_out = dpkt.pcap.Writer(f_out) 
    
    for ts, buf in pcap_in:
        try:
            eth = dpkt.ethernet.Ethernet(buf)
            ip = eth.data
            udp = ip.data
            dns = dpkt.dns.DNS(udp.data)
            
            if dns.id in ids_legitimos:
                pcap_out.writepkt(buf, ts)
        except:
            continue