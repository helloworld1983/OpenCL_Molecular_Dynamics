import subprocess
import sys
import pprint
import re
import os

class Test_Runner:

	def __init__(self, method_name, potential = ""):
		self.method_name = method_name
		self.results_time = dict()
		self.kernel_time = dict()
		self.results_energy = dict()
		self.potential = potential
		self.iters = 3
		self.particles = [
			16,
			32,
			64,
			128,
			256,
			512,
			1024
			]
		if method_name == "Mol_dyn":
			self.run_prefix = "MDHost_"
		elif method_name == "Monte-Carlo":
			self.run_prefix = "MCHost_"
		else:
			sys.exit("wrong method name")

	def prepare_test(self, parameters):
		cmd_line_rm = "rm -f ../%s/include/parameters.h" % self.method_name
		args = cmd_line_rm.split()
		proc = subprocess.Popen(args, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
		stdout, stderr = proc.communicate()
		if stderr:
			print stderr
			sys.exit("unable to rm parameters.h")
		cmd_line_cp = "cp %s ../%s/include/parameters.h" % (parameters, self.method_name)
		args = cmd_line_cp.split()
		proc = subprocess.Popen(args, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
		stdout, stderr = proc.communicate()
		if stderr:
			print stderr
			sys.exit("unable to cp parameters.h")

	def build_all(self):
		cmd_line_make_cpu = "make cpu"
		args = cmd_line_make_cpu.split()
		proc = subprocess.Popen(args, stdout = subprocess.PIPE, stderr = subprocess.PIPE, cwd = "../%s" % self.method_name)
		stdout, stderr = proc.communicate()
		if stderr:
			print stderr
			sys.exit("unable to build cpu %s" % self.method_name)
		cmd_line_make_nvidia_gpu = "make nvidia_gpu"
		args = cmd_line_make_nvidia_gpu.split()
		proc = subprocess.Popen(args, stdout = subprocess.PIPE, stderr = subprocess.PIPE, cwd = "../%s" % self.method_name)
		stdout, stderr = proc.communicate()
		if stderr:
			print stderr
			sys.exit("unable to build nvidia gpu %s" % self.method_name)
		cmd_line_make_intel_gpu = "make intel_gpu"
		args = cmd_line_make_intel_gpu.split()
		proc = subprocess.Popen(args, stdout = subprocess.PIPE, stderr = subprocess.PIPE, cwd = "../%s" % self.method_name)
		stdout, stderr = proc.communicate()
		if stderr:
			print stderr
			sys.exit("unable to build intel gpu %s" % self.method_name)

	def run_tests(self):
		cmd_line_run_cpu = "%sCPU.exe" % (self.run_prefix)
		cmd_line_run_gpu = "%sGPU.exe" % (self.run_prefix)
		cmd_line_run_iocl = "%sIOCL.exe" % (self.run_prefix)
		self.results_energy["CPU"] = dict()
		self.results_time["CPU"] = dict()
		self.results_energy["GPU"] = dict()
		self.kernel_time["GPU"] = dict()
		self.results_time["GPU"] = dict()
		self.results_energy["IOCL"] = dict()
		self.results_time["IOCL"] = dict()
		self.kernel_time["IOCL"] = dict()
		for count in self.particles:
			self.prepare_test(os.path.join(os.path.dirname(os.path.realpath(__file__)), self.method_name, str(count) + "_" +  self.method_name + "_parameters.h"))
			self.build_all()
			for it in range(0,self.iters):
				proc = subprocess.Popen(["../%s/" % (self.method_name) + cmd_line_run_cpu, self.potential], stdout = subprocess.PIPE, stderr = subprocess.PIPE)
				stdout, stderr = proc.communicate()
				if not stderr:
					res = stdout.split("\n")
					print res
					try:
						self.results_energy["CPU"][count].append(re.findall("[-+]?\d+[\.]?\d*[eE]?[-+]?\d*", res[0]))
					except:
						self.results_energy["CPU"][count] = list()
						self.results_energy["CPU"][count].append(re.findall("[-+]?\d+[\.]?\d*[eE]?[-+]?\d*", res[0]))
					try:
						self.results_time["CPU"][count].append(re.findall("[-+]?\d+[\.]?\d*[eE]?[-+]?\d*", res[2]))
					except:
						self.results_time["CPU"][count] = list()
						self.results_time["CPU"][count].append(re.findall("[-+]?\d+[\.]?\d*[eE]?[-+]?\d*", res[2]))
				else:
					print stderr

				proc = subprocess.Popen(["../%s/" % (self.method_name) + cmd_line_run_gpu, self.potential], stdout = subprocess.PIPE, stderr = subprocess.PIPE, cwd = "../%s" % self.method_name)
				stdout, stderr = proc.communicate()
				if not stderr:
					res = stdout.split("\n")
					print res
					try:
						self.results_energy["GPU"][count].append(re.findall("[-+]?\d+[\.]?\d*[eE]?[-+]?\d*", res[1]))
					except:
						self.results_energy["GPU"][count] = list()
						self.results_energy["GPU"][count].append(re.findall("[-+]?\d+[\.]?\d*[eE]?[-+]?\d*", res[1]))
					try:
						self.results_time["GPU"][count].append(re.findall("[-+]?\d+[\.]?\d*[eE]?[-+]?\d*", res[3]))
					except:
						self.results_time["GPU"][count] = list()
						self.results_time["GPU"][count].append(re.findall("[-+]?\d+[\.]?\d*[eE]?[-+]?\d*", res[3]))
					try:
						self.kernel_time["GPU"][count].append(re.findall("[-+]?\d+[\.]?\d*[eE]?[-+]?\d*", res[5]))
					except:
						self.kernel_time["GPU"][count] = list()
						self.kernel_time["GPU"][count].append(re.findall("[-+]?\d+[\.]?\d*[eE]?[-+]?\d*", res[5]))
				else:
					print stderr

				proc = subprocess.Popen(["../%s/" % (self.method_name) + cmd_line_run_iocl, self.potential], stdout = subprocess.PIPE, stderr = subprocess.PIPE, cwd = "../%s" % self.method_name)
				stdout, stderr = proc.communicate()
				if not stderr:
					res = stdout.split("\n")
					print res
					try:
						self.results_energy["IOCL"][count].append(re.findall("[-+]?\d+[\.]?\d*[eE]?[-+]?\d*", res[1]))
					except:
						self.results_energy["IOCL"][count] = list()
						self.results_energy["IOCL"][count].append(re.findall("[-+]?\d+[\.]?\d*[eE]?[-+]?\d*", res[1]))
					try:
						self.results_time["IOCL"][count].append(re.findall("[-+]?\d+[\.]?\d*[eE]?[-+]?\d*", res[3]))
					except:
						self.results_time["IOCL"][count] = list()
						self.results_time["IOCL"][count].append(re.findall("[-+]?\d+[\.]?\d*[eE]?[-+]?\d*", res[3]))
					try:
						self.kernel_time["IOCL"][count].append(re.findall("[-+]?\d+[\.]?\d*[eE]?[-+]?\d*", res[5]))
					except:
						self.kernel_time["IOCL"][count] = list()
						self.kernel_time["IOCL"][count].append(re.findall("[-+]?\d+[\.]?\d*[eE]?[-+]?\d*", res[5]))
				else:
					print stderr

	def import_results(self):
		print "Energy"
		pprint.pprint(self.results_energy)
		print "Total time"
		pprint.pprint(self.results_time)
		print "Kernel time"
		pprint.pprint(self.kernel_time)

t = Test_Runner("Monte-Carlo")
t.run_tests()
t.import_results()

#t = Test_Runner("Mol_dyn")
#t.run_tests()
#t.import_results()