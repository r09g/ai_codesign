import os

def get_cacti(node, size, bw, port, dirpath):
    cur_dir = os.getcwd()
    os.chdir(dirpath)
    size_cmd = 'sed -i \'1s/.*/' + '-size (bytes) ' + str(size) + '/\' cache.cfg'
    bw_cmd = 'sed -i \'2s/.*/' + '-output\/input bus width ' + str(bw) + '/\' cache.cfg'
    r_port_cmd = 'sed -i \'3s/.*/' + '-exclusive read port ' + str(port) + '/\' cache.cfg'
    w_port_cmd = 'sed -i \'4s/.*/' + '-exclusive write port ' + str(port) + '/\' cache.cfg'
    node_cmd = 'sed -i \'5s/.*/' + '-technology (u) ' + str(node) + '/\' cache.cfg'
    os.system(size_cmd)
    os.system(bw_cmd)
    os.system(r_port_cmd)
    os.system(w_port_cmd)
    os.system(node_cmd)

    if os.path.exists('cache.cfg.out'):
        os.remove('cache.cfg.out')

    os.system('./cacti -infile cache.cfg')

    mem_specs = None
    with open('cache.cfg.out','r') as f:
        f.readline()
        mem_specs = f.readline().split(", ")

    data = {}
    data['node'] = float(mem_specs[0])
    data['size_bytes'] = float(mem_specs[1])
    data['bw'] = float(mem_specs[4])
    data['latency_ns'] = float(mem_specs[5])
    data['cost'] = float(mem_specs[8])/2 + float(mem_specs[9])/2
    data['port'] = int(port)

    os.chdir(cur_dir)
    
    return data