OPENQASM 2.0;
include "qelib1.inc";
qreg q[50];
rz(-pi/2) q[2];
sx q[2];
rz(-pi/2) q[2];
cx q[2],q[1];
cx q[1],q[0];
rz(-pi) q[1];
x q[1];
x q[2];
cx q[2],q[1];
cx q[0],q[3];
rz(-pi) q[0];
x q[0];
cx q[1],q[0];
cx q[3],q[4];
rz(-pi) q[3];
x q[3];
cx q[0],q[3];
cx q[4],q[5];
rz(-pi) q[4];
x q[4];
cx q[3],q[4];
cx q[5],q[6];
rz(-pi) q[5];
x q[5];
cx q[4],q[5];
cx q[6],q[7];
rz(-pi) q[6];
x q[6];
cx q[5],q[6];
cx q[7],q[12];
cx q[12],q[11];
cx q[11],q[10];
rz(-pi) q[11];
x q[11];
rz(-pi) q[12];
x q[12];
rz(-pi) q[7];
x q[7];
cx q[6],q[7];
cx q[7],q[12];
cx q[12],q[11];
cx q[10],q[13];
rz(-pi) q[10];
x q[10];
cx q[11],q[10];
cx q[13],q[14];
rz(-pi) q[13];
x q[13];
cx q[10],q[13];
cx q[14],q[15];
rz(-pi) q[14];
x q[14];
cx q[13],q[14];
cx q[15],q[16];
rz(-pi) q[15];
x q[15];
cx q[14],q[15];
cx q[16],q[17];
rz(-pi) q[16];
x q[16];
cx q[15],q[16];
cx q[17],q[22];
rz(-pi) q[17];
x q[17];
cx q[16],q[17];
cx q[22],q[21];
cx q[21],q[20];
rz(-pi) q[21];
x q[21];
rz(-pi) q[22];
x q[22];
cx q[17],q[22];
cx q[22],q[21];
cx q[20],q[23];
rz(-pi) q[20];
x q[20];
cx q[21],q[20];
cx q[23],q[24];
rz(-pi) q[23];
x q[23];
cx q[20],q[23];
cx q[24],q[25];
cx q[23],q[24];
cx q[25],q[26];
cx q[24],q[25];
cx q[26],q[27];
cx q[25],q[26];
cx q[27],q[32];
cx q[26],q[27];
cx q[32],q[31];
cx q[27],q[32];
cx q[31],q[30];
cx q[32],q[31];
cx q[30],q[33];
cx q[31],q[30];
cx q[33],q[34];
cx q[30],q[33];
cx q[34],q[35];
cx q[33],q[34];
cx q[35],q[36];
cx q[34],q[35];
cx q[36],q[37];
cx q[35],q[36];
cx q[37],q[42];
cx q[36],q[37];
cx q[42],q[41];
cx q[37],q[42];
cx q[41],q[40];
cx q[42],q[41];
cx q[40],q[43];
cx q[41],q[40];
cx q[43],q[44];
cx q[40],q[43];
cx q[44],q[45];
cx q[43],q[44];
cx q[45],q[46];
cx q[44],q[45];
cx q[46],q[49];
cx q[45],q[46];
cx q[46],q[49];
