//
//  List_Files.cpp
//  
//
//  Created by Rohit Shukla on 9/1/15.
//
//

#include <stdio.h>
#include <iostream>
#include "dirent.h"

using namespace std;

int main(){
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir ("./Caltech-101/Segmented_Image/tick")) != NULL) {
        /* print all the files and directories within directory */
        while ((ent = readdir (dir)) != NULL) {
            printf ("%s\n", ent->d_name);
        }
        closedir (dir);
    } else {
        /* could not open directory */
        perror ("");
        return EXIT_FAILURE;
    }
    return 0;
}
