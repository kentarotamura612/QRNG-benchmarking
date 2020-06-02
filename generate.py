#!/Users/kentaro/anaconda3/bin/python3

from qiskit import IBMQ, execute, QuantumRegister, ClassicalRegister, QuantumCircuit, Aer
from qiskit.compiler import transpile
import sys
import argparse
from copy import deepcopy
import time
import pandas as pd

def get_backends(hub='ibm-q-keio', group='keio-internal', project='keio-students'):
    """
        For internal ibm/trl hub='ibm-q-internal', group='trl', project='qiskit'
    """
    IBMQ.load_account()
    backends = dict()
    my_provider = IBMQ.get_provider(hub=hub, group=group, project=project)
    pub_provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
    for each in my_provider.backends(simulator=False, operational=True):
        backends[each.name()] = my_provider.get_backend(each.name())
    for each in pub_provider.backends(simulator=False, operational=True):
        backends[each.name()] = pub_provider.get_backend(each.name())
    return backends

def get_args():
    parser = argparse.ArgumentParser(description="generate random numbers with Hadamard gates on IBMQ devices")
    parser.add_argument("-d", "--dir", help="the name of directory to store files", default=None)
    parser.add_argument("-j", "--jobs", help="number of jobs to run", default=1, type=int)
    parser.add_argument("-b", "--backend", help="name of backend to run", type=str)
    parser.add_argument("-c", "--circuits", help="number of circuits per job", type=int, default=1)
    parser.add_argument("-s", "--shots", help="number of shots", default=8192)
    parser.add_argument("-g", "--group", help="group of ibm q device", default="keio-internal", type=str)
    parser.add_argument("-p", "--project", help="project of ibm q device", default="keio-students", type=str)
    parser.add_argument("-u", "--hub", help="hub of ibm q device", default="ibm-q-keio", type=str)
    return parser.parse_args()

def create_h_circuit(backend, ncircuits = 1):
    """
        Create a circuit with H gates for random 0-1 sequences
        Arguments:
            backend: backend used
            isX: if True append X gates before measurement at all qubits
        RETURN:
            transpiled circuits
    """
    NQBITS = backend.configuration().n_qubits
    cmap = backend.configuration().coupling_map
    qr = QuantumRegister(NQBITS, "qr")
    cr = ClassicalRegister(NQBITS, "cr")
    qc = QuantumCircuit(qr, cr)

    #for devices with more than 5 qubits,
    for i in range(NQBITS):
        qc.h(qr[i])
    for i in range(NQBITS):
        qc.measure(qr[i], cr[i])

    qc = transpile(qc, backend=backend, optimization_level=0)
    circuits = []
    for i in range(ncircuits):
        circuits.append(deepcopy(qc))
    return circuits

def run_job(circuits, backend, dir, samples, shots=8192):
    jobids = []
    NQBITS = backend.configuration().n_qubits
    ncircuits = len(circuits)
    job = execute(circuits, backend, shots=shots, memory=True)
    for i in range(ncircuits):
        jobids.append(job.job_id())

    jobid = jobids[-1]
    device = backend
    device_name = device.name()
    properties = device.properties()

    #save properties, config, noise_model to assigned folders
    pd.to_pickle(properties, f'{dir}/properties/{jobid}_prop_{device_name}.dat')

    #execute circuit
    result = job.result()

    #save job_id
    pd.to_pickle(jobids, f'{dir}/jobids/id_{device_name}.dat')

    #obtain memory sequence
    for n in range(ncircuits):
        sequence = result.get_memory(n)
        pd.to_pickle(sequence, f'{dir}/sequence/{jobid}_{n}_{device_name}seq.dat')
        samples.append(f'{dir}/sequence/{jobid}_{n}_{device_name}seq.dat')
        print(f"job done at {device_name} ID: {jobid} #{n+1}")

    return samples

def get_curr_dir():
    import os
    return os.getcwd().rstrip("/")


def create_dir(dir):
    import os
    try:
        if not os.path.exists(dir):
            os.mkdir(dir)
        if not os.path.exists(dir+"/properties"):
            os.mkdir(dir+"/properties")
        if not os.path.exists(dir+"/sequence"):
            os.mkdir(dir+"/sequence")
        if not os.path.exists(dir+"/jobids"):
            os.mkdir(dir+"/jobids")
        if not os.path.exists(dir+"/paths"):
            os.mkdir(dir+"/paths")
    except FileExistsError:
        pass


if __name__ == "__main__":

    args = get_args()
    #print(args.dir, args.jobs, args.backend, args.withX, args.circuits, args.offline)
    if args.dir is None:
        #print(get_curr_dir())
        args.dir = get_curr_dir()
    else:
        args.dir = get_curr_dir() + "/" + args.dir
        create_dir(args.dir)

    backends = get_backends(args.hub, args.group, args.project)
    if args.backend is None or args.backend not in backends:
        print("Please specify backend as one of the followings:")
        for b in backends:
            print("\t", b)
        sys.exit(1)

    backend = backends[args.backend]
    circuits = create_h_circuit(backend, args.circuits)
    samples = []
    for j in range(args.jobs):
        sample_pass = run_job(circuits, backend, args.dir, samples, args.shots)
        #save sample pass
    pd.to_pickle(sample_pass, f'{args.dir}/paths/path_{backend}.dat')
    exit("done")
