OPENQASM 2.0;
include "qelib1.inc";
qreg q[70];
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
cx q[3],q[0];
x q[0];
cx q[0],q[1];
sx q[0];
rz(3*pi/4) q[0];
sx q[0];
rz(pi/2) q[0];
x q[1];
cx q[1],q[2];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
sx q[3];
rz(-3*pi/4) q[3];
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
rz(pi/2) q[7];
cx q[7],q[6];
x q[6];
cx q[6],q[5];
x q[5];
cx q[5],q[4];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(-pi/2) q[6];
sx q[6];
rz(-3*pi/4) q[6];
sx q[6];
rz(pi/2) q[6];
sx q[7];
rz(3*pi/4) q[7];
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
cx q[13],q[10];
x q[10];
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
rz(-3*pi/4) q[13];
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
cx q[17],q[16];
rz(-pi) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[16],q[15];
x q[15];
cx q[15],q[14];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
sx q[16];
rz(3*pi/4) q[16];
sx q[16];
rz(pi/2) q[16];
sx q[17];
rz(3*pi/4) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/2) q[21];
sx q[21];
rz(-3*pi/4) q[21];
sx q[21];
rz(pi/2) q[21];
rz(pi/2) q[22];
sx q[22];
rz(-3*pi/4) q[22];
sx q[22];
rz(pi/2) q[22];
rz(pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
cx q[23],q[20];
x q[20];
cx q[20],q[21];
rz(-pi/2) q[20];
sx q[20];
rz(-3*pi/4) q[20];
sx q[20];
rz(pi/2) q[20];
rz(-pi) q[21];
sx q[21];
rz(pi) q[21];
cx q[21],q[22];
sx q[21];
rz(pi/2) q[21];
sx q[23];
rz(3*pi/4) q[23];
sx q[23];
rz(pi/2) q[23];
rz(pi/2) q[24];
sx q[24];
rz(-3*pi/4) q[24];
sx q[24];
rz(pi/2) q[24];
rz(pi/2) q[25];
sx q[25];
rz(-3*pi/4) q[25];
sx q[25];
rz(pi/2) q[25];
rz(pi/2) q[27];
sx q[27];
cx q[27],q[26];
x q[26];
cx q[26],q[25];
x q[25];
cx q[25],q[24];
rz(pi/2) q[25];
sx q[25];
rz(pi/2) q[25];
rz(-pi/2) q[26];
sx q[26];
rz(-3*pi/4) q[26];
sx q[26];
rz(pi/2) q[26];
sx q[27];
rz(-3*pi/4) q[27];
sx q[27];
rz(pi/2) q[27];
rz(pi/2) q[31];
sx q[31];
rz(-3*pi/4) q[31];
sx q[31];
rz(pi/2) q[31];
rz(pi/2) q[32];
sx q[32];
rz(-3*pi/4) q[32];
sx q[32];
rz(pi/2) q[32];
rz(pi/2) q[33];
sx q[33];
cx q[33],q[30];
x q[30];
cx q[30],q[31];
rz(-pi/2) q[30];
sx q[30];
rz(-3*pi/4) q[30];
sx q[30];
rz(pi/2) q[30];
x q[31];
cx q[31],q[32];
rz(pi/2) q[31];
sx q[31];
rz(pi/2) q[31];
sx q[33];
rz(-3*pi/4) q[33];
sx q[33];
rz(pi/2) q[33];
rz(pi/2) q[34];
sx q[34];
rz(-3*pi/4) q[34];
sx q[34];
rz(pi/2) q[34];
rz(pi/2) q[35];
sx q[35];
rz(-3*pi/4) q[35];
sx q[35];
rz(pi/2) q[35];
rz(pi/2) q[37];
sx q[37];
rz(pi/2) q[37];
cx q[37],q[36];
rz(-pi) q[36];
sx q[36];
cx q[36],q[35];
rz(-pi) q[35];
sx q[35];
rz(pi) q[35];
cx q[35],q[34];
sx q[35];
rz(pi/2) q[35];
sx q[36];
rz(-3*pi/4) q[36];
sx q[36];
rz(pi/2) q[36];
sx q[37];
rz(3*pi/4) q[37];
sx q[37];
rz(pi/2) q[37];
rz(pi/2) q[41];
sx q[41];
rz(-3*pi/4) q[41];
sx q[41];
rz(pi/2) q[41];
rz(pi/2) q[42];
sx q[42];
rz(-3*pi/4) q[42];
sx q[42];
rz(pi/2) q[42];
rz(pi/2) q[43];
sx q[43];
cx q[43],q[40];
rz(-pi) q[40];
sx q[40];
cx q[40],q[41];
sx q[40];
rz(-3*pi/4) q[40];
sx q[40];
rz(pi/2) q[40];
x q[41];
cx q[41],q[42];
rz(pi/2) q[41];
sx q[41];
rz(pi/2) q[41];
sx q[43];
rz(-3*pi/4) q[43];
sx q[43];
rz(pi/2) q[43];
rz(pi/2) q[44];
sx q[44];
rz(-3*pi/4) q[44];
sx q[44];
rz(pi/2) q[44];
rz(pi/2) q[45];
sx q[45];
rz(-3*pi/4) q[45];
sx q[45];
rz(pi/2) q[45];
rz(pi/2) q[47];
sx q[47];
rz(pi/2) q[47];
cx q[47],q[46];
x q[46];
cx q[46],q[45];
x q[45];
cx q[45],q[44];
rz(pi/2) q[45];
sx q[45];
rz(pi/2) q[45];
sx q[46];
rz(3*pi/4) q[46];
sx q[46];
rz(pi/2) q[46];
sx q[47];
rz(3*pi/4) q[47];
sx q[47];
rz(pi/2) q[47];
rz(pi/2) q[51];
sx q[51];
rz(-3*pi/4) q[51];
sx q[51];
rz(pi/2) q[51];
rz(pi/2) q[52];
sx q[52];
rz(-3*pi/4) q[52];
sx q[52];
rz(pi/2) q[52];
rz(pi/2) q[53];
sx q[53];
cx q[53],q[50];
rz(-pi) q[50];
sx q[50];
cx q[50],q[51];
sx q[50];
rz(-3*pi/4) q[50];
sx q[50];
rz(pi/2) q[50];
rz(-pi) q[51];
sx q[51];
rz(pi) q[51];
cx q[51],q[52];
sx q[51];
rz(pi/2) q[51];
sx q[53];
rz(-3*pi/4) q[53];
sx q[53];
rz(pi/2) q[53];
rz(pi/2) q[54];
sx q[54];
rz(-3*pi/4) q[54];
sx q[54];
rz(pi/2) q[54];
rz(pi/2) q[55];
sx q[55];
rz(-3*pi/4) q[55];
sx q[55];
rz(pi/2) q[55];
rz(pi/2) q[57];
sx q[57];
rz(pi/2) q[57];
rz(pi/2) q[59];
sx q[59];
rz(pi/2) q[59];
cx q[59],q[56];
rz(-pi) q[56];
sx q[56];
cx q[56],q[55];
rz(-pi) q[55];
sx q[55];
rz(pi) q[55];
cx q[55],q[54];
sx q[55];
rz(pi/2) q[55];
sx q[56];
rz(-3*pi/4) q[56];
sx q[56];
rz(pi/2) q[56];
sx q[59];
rz(3*pi/4) q[59];
sx q[59];
rz(pi/2) q[59];
rz(pi/2) q[60];
sx q[60];
rz(-3*pi/4) q[60];
sx q[60];
rz(pi/2) q[60];
rz(pi/2) q[61];
sx q[61];
rz(-3*pi/4) q[61];
sx q[61];
rz(pi/2) q[61];
cx q[57],q[62];
sx q[57];
rz(3*pi/4) q[57];
sx q[57];
rz(pi/2) q[57];
x q[62];
cx q[62],q[61];
x q[61];
cx q[61],q[60];
rz(pi/2) q[61];
sx q[61];
rz(pi/2) q[61];
rz(-pi/2) q[62];
sx q[62];
rz(-3*pi/4) q[62];
sx q[62];
rz(pi/2) q[62];
rz(pi/2) q[63];
sx q[63];
rz(pi/2) q[63];
cx q[63],q[64];
sx q[63];
rz(3*pi/4) q[63];
sx q[63];
rz(pi/2) q[63];
x q[64];
rz(pi/2) q[65];
sx q[65];
rz(-3*pi/4) q[65];
sx q[65];
rz(pi/2) q[65];
cx q[64],q[65];
rz(-pi/2) q[64];
sx q[64];
rz(-3*pi/4) q[64];
sx q[64];
rz(pi/2) q[64];
rz(-pi) q[65];
sx q[65];
rz(pi) q[65];
rz(pi/2) q[66];
sx q[66];
rz(-3*pi/4) q[66];
sx q[66];
rz(pi/2) q[66];
cx q[65],q[66];
sx q[65];
rz(pi/2) q[65];