# Temporal correlation

Ideally, quantum computers should be capable of generating unbiased and independent random number sequences. However, this is not necessarily the case with current NISQ devices [1, 2, 3]. The random number sequences generated by real quantum computers contain bias and dependence [1]. The temporal correlation project allows IBMQ users to generate random number sequences with any IBMQ device and assess the resulting sequences in terms of temporal correlation. Please refer to our paper [1] in the reference section for further theoretical information.

# Description
The temporal correlation project consists of the following 3 programs.
```
generate.py - generates multiple random number sequences using all qubits of the IBMQ device specified by the user.
to_text.py - converts the generated random number sequences into text files per qubit.
plot.py - computes the p-values corresponding to each random number sequence per qubit and creates a figure as below.
```
The goal is to visualize the temporal correlation in the random number sequences generated by the qubit contained in the IBMQ devices.
# Requirements

Below are the library versions under which the author has tested the program.

```
Python                             3.7.3
matplotlib                         3.1.0
numpy                              1.18.4
pandas                             0.24.2
qiskit                             0.19.1
qiskit-aer                         0.5.1
qiskit-ibmq-provider               0.7.0
qiskit-terra                       0.14.1
scipy                              1.4.1
```

# Usage

The input of IBMQ devices are called jobs. Each job can contain several quantum circuits, and each quantum circuit is run repeatedly. The number of times each circuit is run is called shots. `generate.py` has 8 parameters that the user needs to specify.

```
-d [the directory under which the results will be stored]
-b [the name of the IBMQ device to be used]
-j [the number of jobs to be run]
-c [the number of circuits to be contained in each job]
-s [the number of times each circuit is run (default: 8,192)]
-u [the hub name of your IBMQ account]
-g [the group name of your IBMQ account]
-p [the project name of your IBMQ account]
```
Consider the case where one decided to run 2 jobs, each job consisting of 10 random number generation circuits. This should yield 2 * 10 = 20 sequences per qubit. Under the default setting, which is set at the maximum number of shots on any IBMQ device, each sequence is 8,192 bits long.
```
python3 generate.py -d test -b ibmq_cambridge -c 10 -j 2 -u ibm-q-keio -g keio-internal -p keio-students
```
Upon execution, `generate.py` yields 3 folders under the specified directory, which in this case is `test`.
```
jobids - where the job ids are stored.
properties - where the properties of the device at the time the job was sent is stored.
pass- where the paths for each sequence is stored in chronological order
sequence - where the random number sequences are stored.
```
Next, execute `to_text.py`, which converts the random number sequences obtained into text files per qubit. Don't forget to specify the same directory `generate.py` was executed under.
```
python3 to_text.py -d test
```
This should yield a folder called `text` where the random number sequences output by each qubit are stored in the form of .txt files. Finally, run `plot.py` to compute the temporal correlation p-values for each sequence of each qubit to obtain a figure.
```
python3 plot.py -d test
```
`plot.py` creates 2 folders.
```
autocorr - where the p-values and test statistic values are stored.
figures - where the temporal correlation figure is stored.
```

# References
[1] Y. Shikano, K. Tamura and R. Raymond, "Detecting Temporal Correlation via Quantum Random Number Generation", EPTCS 315, 18-25 (2020). (http://eptcs.web.cse.unsw.edu.au/paper.cgi?QSQW19.2)

[2] K. Tamura and Y. Shikano, "Quantum Random Numbers generated by the Cloud Superconducting Quantum Computer", arXiv:1906.04410 (2020). (https://arxiv.org/abs/1906.04410) to be published in International Symposium on Mathematics, Quantum Theory, and Cryptography: Proceedings of MQC 2019 edited by Tsuyoshi Takagi, Masato Wakayama, Keisuke Tanaka, Noboru Kunihiro, Kazufumi Kimoto, and Yasuhiko Ikematsu (Springer Nature, Singapore, 2020).

[3] K. Tamura and Y. Shikano, "Quantum Random Number Generation with the Superconducting Quantum Computer IBM 20Q Tokyo", in Proceedings of Workshop on Quantum Computing and Quantum Information, edited by Mika Hirvensalo and Abuzer Yakaryilmaz, TUCS Lecture Notes 30, 13-25 (2019).
(https://www.utupub.fi/handle/10024/147810); Cryptology ePrint Archive: Report 2020/078 (https://eprint.iacr.org/2020/078)
