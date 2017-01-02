#ifndef LOGGER
	#define LOGGER
	static char log_file[32] = "results.txt";
	static int print_time_line = 0;
	void log_print(char* filename, int line, char *fmt,...);
	#define LOG_PRINT(...) log_print(__FILE__, __LINE__, __VA_ARGS__ )

	#include <stdio.h>
	#include <stdarg.h>
	#include <time.h>
	#include <string.h>
	#include <stdlib.h>

	FILE *fp ;
	static int SESSION_TRACKER;

	char* print_time(){
	    int size = 0;
	    time_t t;
	    char *buf;
	    t=time(NULL);
	    char *timestr = asctime(localtime(&t));
	    timestr[strlen(timestr) - 1] = 0;
	    size = sizeof(char) * (strlen(timestr)+ 1 + 2);
	    buf = (char*)malloc(size);
	    memset(buf, 0x0, size);
	    snprintf(buf,size,"[%s]", timestr);
	    return buf;
	}

	void log_print(char* filename, int line, char *fmt, ...){
	    va_list list;
	    char *p, *string_value;
	    int int_value;
	    double float_value;

	    if(SESSION_TRACKER > 0)
	      fp = fopen (log_file,"a+");
	    else
	      fp = fopen (log_file,"w");

	    if (print_time_line){
	            fprintf(fp,"%s ", print_time());
	            fprintf(fp,"[%s][line: %d] ",filename,line);
	    }
	    va_start( list, fmt );

	    for (p = fmt ; *p ; ++p){
	        if ( *p != '%' ){
	            fputc( *p,fp );
	        }
	        else{
	            switch ( *++p ){
	            //string
	            case 's':{
	                string_value = va_arg(list, char*);
	                fprintf(fp,"%s", string_value);
	                continue;
	            }
	            //int
	            case 'd':{
	                int_value = va_arg(list, int);
	                fprintf(fp,"%d", int_value);
	                continue;
	            }
	            //float
	            case 'f':{
	                float_value = va_arg(list, double);
	                fprintf(fp,"%f", float_value);
	                continue;
	            }
	            default:
	                fputc( *p, fp );
	            }
	        }
	    }
	    va_end( list );
	    fputc( '\n', fp );
	    SESSION_TRACKER++;
	    fclose(fp);
	}
#endif