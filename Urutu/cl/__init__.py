# Developed by Aditya Atluri
# Date: 18 Jan 2014
# Mail: pyurutu@gmail.com
# This file contains the OpenCL implementation of the Python Code

import inspect,shlex
import numpy as np
import execl

class cl_test:
	arguments=[]
	returns=[]
	variables_nam=[]
	variables_val=[]
	kernel="/**/"
	threads=[]
	blocks=[]
	threads_dec=False
	blocks_dec=False
	func_name=[]
	code=""
	args = []

	def __init__(self,fn,args):
		stri=inspect.getsource(fn)
		sh=shlex.shlex(stri)
		self.code=stri
		self.args = args

	def print_funcname_cl(self,def_i,a):
		self.func_name=a[def_i+1]
		self.kernel=self.kernel+"__kernel void CL_kernel("
		return def_i+2

	def body_dev(self,stri_i):
		self.kernel=self.kernel+stri_i+";\n"

	def inspectit(self,stri_1):
		sh=shlex.shlex(stri_1)
		i=sh.get_token()
		if i=='def' or i=='@' or i=='return' or i=='' or i=='#':
			return
		stmt=[]
		while i is not sh.eof:
			stmt.append(i)
			i=sh.get_token()
		if self.threads_dec==False:
			self.threads_decl(stmt)
			return
		if self.blocks_dec==False:
			self.blocks_decl(stmt)
			return
		cl_test.body_dev(self,stri_1)
	
	def print_numtabs(self,stri_3):
		for i in stri_3:
			if i.split('\t')!=-1:
				j= i.split('\t')
				cl_test.inspectit(self,j[-1])

	def threads_decl(self,stmt):
		equ=stmt.index('=')
		if self.variables_nam.count('tx')<1&stmt.count('tx')==1:
			pos=stmt.index('tx')
			pos_val=stmt[pos+1+equ]
			self.variables_nam.append(stmt[pos])
			self.variables_val.append(int(pos_val))
			stri_i="int tx = get_global_id(0);\n"
			self.threads.append(2*int(pos_val))			#Threads doubled Here! Just Here!
			self.kernel=self.kernel+stri_i
		if self.variables_nam.count('ty')<1&stmt.count('ty')==1:
			pos=stmt.index('ty')
			pos_val=stmt[pos+1+equ]
			self.variables_nam.append(stmt[pos])
			self.variables_val.append(int(pos_val))
			stri_i="int ty = get_global_id(1);\n"
			self.threads.append(int(pos_val))
			self.kernel=self.kernel+stri_i
		if self.variables_nam.count('tz')<1&stmt.count('tz')==1:
			pos=stmt.index('tz')
			pos_val=stmt[pos+1+equ]
			self.variables_nam.append(stmt[pos])
			self.variables_val.append(int(pos_val))
			stri_i="int tz = get_global_id(2);\n"
			self.threads.append(int(pos_val))
			self.kernel=self.kernel+stri_i
		if len(self.threads)==3:
			self.threads_dec=True

	def blocks_decl(self,stmt):
		equ=stmt.index('=')
		if self.variables_nam.count('bx')<1&stmt.count('bx')==1:
			pos=stmt.index('bx')
			pos_val=stmt[pos+1+equ]
			self.variables_nam.append(stmt[pos])
			self.variables_val.append(int(pos_val))
			stri_i="int bx = get_local_id(0);\n"
			self.blocks.append(int(pos_val))
			self.kernel=self.kernel+stri_i
		if self.variables_nam.count('by')<1&stmt.count('by')==1:
			pos=stmt.index('by')
			pos_val=stmt[pos+1+equ]
			self.variables_nam.append(stmt[pos])
			self.variables_val.append(int(pos_val))
			stri_i="int by = get_local_id(1);\n"
			self.blocks.append(int(pos_val))
			self.kernel=self.kernel+stri_i
		if self.variables_nam.count('bz')<1&stmt.count('bz')==1:
			pos=stmt.index('bz')
			pos_val=stmt[pos+1+equ]
			self.variables_nam.append(stmt[pos])
			self.variables_val.append(int(pos_val))
			stri_i="int bz = get_local_id(2);\n"
			self.blocks.append(int(pos_val))
			self.kernel=self.kernel+stri_i
		if len(self.blocks)==3:
			self.blocks_dec=True

	def print_variables(self,comma,var_i,a):
		if self.arguments.count(a[var_i])<2:
			self.arguments.append(a[var_i])
			if comma==True:
				stri=", __global int* "+a[var_i]
				self.kernel=self.kernel+stri
			if comma==False:
				stri=" __global int* "+a[var_i]
				self.kernel=self.kernel+stri

	def execute(self):
		sh=shlex.shlex(self.code)
		i=sh.get_token()
		a=[i]
		while i is not sh.eof:
			i=sh.get_token()
			a.append(i)
		control=a.index('def')
		control=cl_test.print_funcname_cl(self,control,a)
		comma=False
		if a[control]=='(':
			control=control+1
			while a[control]!=')':
				if a[control]==',':
					control=control+1
				cl_test.print_variables(self,comma,control,a)
				comma=True
				control=control+1
			ret=a.index('return')+1
			while a[ret]!='':
				if a[ret]==',':
					ret=ret+1
				self.returns.append(a[ret])
				ret=ret+1
			self.kernel=self.kernel+"){\n"
			control=control+1
		if a[control]==':':
			control=control+1
		stri_1=self.code.split("\n")
		cl_test.print_numtabs(self,stri_1)
		self.kernel = self.kernel + "}"
#		self.print_cl()
#		for i in self.args:
#			print i
		tmp = execl.cl_exe()
		return tmp.exe_cl(self.kernel,self.func_name,self.threads,self.blocks,self.args,self.returns)

	def print_cl(self):
		print "In print_cl:"
		print self.arguments
		print self.returns
		print self.variables_nam
		print self.variables_val
		print self.kernel
		print self.threads
		print self.blocks
		print self.threads_dec
		print self.blocks_dec
		print self.func_name
		print self.code

def urutu_cl(fn):
	def inner(*args,**kargs):
		cl=cl_test(fn,args)
		return cl.execute()
	return inner
