file(READ "${INPUT}" contents HEX)
string(REGEX REPLACE "(..)" "0x\\1," contents "${contents}")
file(WRITE "${OUTPUT}" "static const unsigned char mracleRunnerObject[] = {${contents}};\n")
