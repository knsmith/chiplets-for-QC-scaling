OPENQASM 2.0;
include "qelib1.inc";
qreg q[30];
rz(pi/2) q[1];
sx q[1];
rz(-3*pi/4) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[2];
sx q[2];
rz(-3*pi/4) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[3],q[0];
x q[0];
cx q[0],q[1];
rz(-pi/2) q[0];
sx q[0];
rz(-3*pi/4) q[0];
sx q[0];
rz(pi/2) q[0];
x q[1];
cx q[1],q[2];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
sx q[3];
rz(3*pi/4) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[4];
sx q[4];
rz(-3*pi/4) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[5];
sx q[5];
rz(-3*pi/4) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[7];
sx q[7];
cx q[7],q[6];
rz(-pi) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[6],q[5];
x q[5];
cx q[5],q[4];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
sx q[6];
rz(3*pi/4) q[6];
sx q[6];
rz(pi/2) q[6];
sx q[7];
rz(-3*pi/4) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[11];
sx q[11];
rz(-3*pi/4) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[12];
sx q[12];
rz(-3*pi/4) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[13],q[10];
rz(-pi) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[10],q[11];
sx q[10];
rz(3*pi/4) q[10];
sx q[10];
rz(pi/2) q[10];
x q[11];
cx q[11],q[12];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
sx q[13];
rz(3*pi/4) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[14];
sx q[14];
rz(-3*pi/4) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[15];
sx q[15];
rz(-3*pi/4) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/2) q[19];
sx q[19];
cx q[19],q[16];
x q[16];
cx q[16],q[15];
rz(-pi) q[15];
sx q[15];
rz(pi) q[15];
cx q[15],q[14];
sx q[15];
rz(pi/2) q[15];
rz(-pi/2) q[16];
sx q[16];
rz(-3*pi/4) q[16];
sx q[16];
rz(pi/2) q[16];
sx q[19];
rz(-3*pi/4) q[19];
sx q[19];
rz(pi/2) q[19];
rz(pi/2) q[20];
sx q[20];
rz(-3*pi/4) q[20];
sx q[20];
rz(pi/2) q[20];
rz(pi/2) q[21];
sx q[21];
rz(-3*pi/4) q[21];
sx q[21];
rz(pi/2) q[21];
cx q[17],q[22];
sx q[17];
rz(3*pi/4) q[17];
sx q[17];
rz(pi/2) q[17];
x q[22];
cx q[22],q[21];
x q[21];
cx q[21],q[20];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(-pi/2) q[22];
sx q[22];
rz(-3*pi/4) q[22];
sx q[22];
rz(pi/2) q[22];
rz(pi/2) q[23];
sx q[23];
cx q[23],q[24];
sx q[23];
rz(-3*pi/4) q[23];
sx q[23];
rz(pi/2) q[23];
rz(-pi) q[24];
sx q[24];
rz(pi/2) q[24];
rz(pi/2) q[25];
sx q[25];
rz(-3*pi/4) q[25];
sx q[25];
rz(pi/2) q[25];
cx q[24],q[25];
sx q[24];
rz(3*pi/4) q[24];
sx q[24];
rz(pi/2) q[24];
x q[25];
rz(pi/2) q[26];
sx q[26];
rz(-3*pi/4) q[26];
sx q[26];
rz(pi/2) q[26];
cx q[25],q[26];
rz(pi/2) q[25];
sx q[25];
rz(pi/2) q[25];
