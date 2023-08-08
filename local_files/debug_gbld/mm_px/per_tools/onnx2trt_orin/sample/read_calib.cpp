

#include <unistd.h>  // For ::getopt
#include <iostream>
#include <fstream>
#include <vector>
#include <string.h> 
#include <stdio.h>

using namespace std;

int main(int argc, char *argv[])
{
    if(argc <= 1) {
        cout<<"Error: Please enter at least one calibration table."<<endl;
        return -1;
    }

    string calibPath = argv[1];
    
    vector<int> scale;
    vector<string> str;
    unsigned int tmp;

    std::ifstream inFile(calibPath);

    if (inFile.good()) {
        string character;
        char str_tmp[100];
        std::getline(inFile, character);
        cout<<"Reading calibration file: "<<character<<endl;

        int iter = 0;
        while (std::getline(inFile, character)) {

            sscanf(character.c_str(), "%s:%*[:^]", str_tmp);
            str.push_back(str_tmp);
            // printf("%s\t", str[iter].c_str());
        
            sscanf(character.c_str(), "%*[^:]:%x", &tmp);
            scale.push_back(tmp);

            iter ++;
        }

        printf("\n");

        float* floatData = reinterpret_cast<float*>(scale.data());

        for(int i = 0; i < scale.size(); ++i)
            // printf("%s\t%.5lf\n", str[i].c_str(), 1.f/ floatData[i]);
            printf("%s\t%.5lf\n", str[i].c_str(), floatData[i]);

        // printf("\n");
            
    }


        
}