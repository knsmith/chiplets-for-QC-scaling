OPENQASM 2.0;
include "qelib1.inc";
qreg q[30];
creg m_mcm0[11];
creg m_meas_all[23];
x q[1];
x q[3];
x q[5];
x q[7];
x q[11];
x q[13];
x q[15];
x q[17];
x q[21];
x q[23];
x q[25];
x q[27];
cx q[27],q[26];
cx q[25],q[26];
cx q[25],q[24];
cx q[23],q[24];
cx q[23],q[20];
cx q[21],q[20];
cx q[21],q[22];
cx q[17],q[22];
cx q[17],q[16];
cx q[15],q[16];
cx q[15],q[14];
cx q[13],q[14];
cx q[13],q[10];
cx q[11],q[10];
cx q[11],q[12];
cx q[7],q[12];
cx q[7],q[6];
cx q[5],q[6];
cx q[5],q[4];
cx q[3],q[4];
cx q[3],q[0];
cx q[1],q[0];
measure q[26] -> m_mcm0[0];
reset q[26];
measure q[24] -> m_mcm0[1];
reset q[24];
measure q[20] -> m_mcm0[2];
reset q[20];
measure q[22] -> m_mcm0[3];
reset q[22];
measure q[16] -> m_mcm0[4];
reset q[16];
measure q[14] -> m_mcm0[5];
reset q[14];
measure q[10] -> m_mcm0[6];
reset q[10];
measure q[12] -> m_mcm0[7];
reset q[12];
measure q[6] -> m_mcm0[8];
reset q[6];
measure q[4] -> m_mcm0[9];
reset q[4];
measure q[0] -> m_mcm0[10];
reset q[0];
measure q[27] -> m_meas_all[0];
measure q[26] -> m_meas_all[1];
measure q[25] -> m_meas_all[2];
measure q[24] -> m_meas_all[3];
measure q[23] -> m_meas_all[4];
measure q[20] -> m_meas_all[5];
measure q[21] -> m_meas_all[6];
measure q[22] -> m_meas_all[7];
measure q[17] -> m_meas_all[8];
measure q[16] -> m_meas_all[9];
measure q[15] -> m_meas_all[10];
measure q[14] -> m_meas_all[11];
measure q[13] -> m_meas_all[12];
measure q[10] -> m_meas_all[13];
measure q[11] -> m_meas_all[14];
measure q[12] -> m_meas_all[15];
measure q[7] -> m_meas_all[16];
measure q[6] -> m_meas_all[17];
measure q[5] -> m_meas_all[18];
measure q[4] -> m_meas_all[19];
measure q[3] -> m_meas_all[20];
measure q[0] -> m_meas_all[21];
measure q[1] -> m_meas_all[22];
