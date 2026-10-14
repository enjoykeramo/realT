@echo off
rem ============================================================================
rem  realT build script (Windows)
rem  Compiles the project and packages a portable app: realt.exe + bundled Java.
rem
rem  Usage:   build.bat              output to  dist\realt
rem           build.bat D:\tools\realt   output to the given folder
rem           build.bat test         run unit tests first, then build
rem
rem  Needs:   JDK 17 or newer (includes jpackage), JAVA_HOME set or java on PATH
rem           Gradle 8.x on PATH (else the bundled gradlew.bat is used)
rem
rem  Files you created in the output folder (model1, out1, ...) are kept.
rem  Only the program parts (app, runtime, realt.exe) are replaced.
rem ============================================================================
setlocal EnableExtensions
set "ROOT=%~dp0"
set "ROOT=%ROOT:~0,-1%"

set "RUNTESTS="
set "OUT="
:args
if "%~1"=="" goto argsdone
if /i "%~1"=="test" (set "RUNTESTS=1") else (set "OUT=%~f1")
shift
goto args
:argsdone
cd /d "%ROOT%"
if not defined OUT set "OUT=%ROOT%\dist\realt"
if "%OUT:~-1%"=="\" set "OUT=%OUT:~0,-1%"

echo.
echo ==== realT build ====
echo Source : %ROOT%
echo Output : %OUT%
echo.

rem ---- 1. Find JDK tools -------------------------------------------------------
set "JPACKAGE="
if defined JAVA_HOME if exist "%JAVA_HOME%\bin\jpackage.exe" set "JPACKAGE=%JAVA_HOME%\bin\jpackage.exe"
if not defined JPACKAGE for /f "delims=" %%i in ('where jpackage.exe 2^>nul') do if not defined JPACKAGE set "JPACKAGE=%%i"
if not defined JPACKAGE (
    echo [ERROR] jpackage not found. Install JDK 17 or newer, then set JAVA_HOME
    echo         to the JDK folder or add its bin folder to PATH.
    goto fail
)
echo [1/4] JDK tools: %JPACKAGE%

rem ---- 2. Gradle build -----------------------------------------------------------
set "GRADLE="
for /f "delims=" %%i in ('where gradle.bat 2^>nul') do if not defined GRADLE set "GRADLE=%%i"
if not defined GRADLE set "GRADLE=%ROOT%\gradlew.bat"
echo [2/4] Gradle: %GRADLE%
set "TASKS=clean :app:installDist"
if defined RUNTESTS set "TASKS=clean test :app:installDist"
call "%GRADLE%" --no-daemon %TASKS%
if errorlevel 1 (
    echo [ERROR] Gradle build failed, see messages above.
    goto fail
)

set "LIB=%ROOT%\app\build\install\realt\lib"
set "MAINJAR="
for %%f in ("%LIB%\app-*.jar") do set "MAINJAR=%%~nxf"
if not defined MAINJAR (
    echo [ERROR] app-*.jar not found in %LIB%
    goto fail
)

rem ---- 3. Package with jpackage (fresh staging folder every time) ---------------
set "STAGE=%ROOT%\build\jpackage"
if exist "%STAGE%" rmdir /s /q "%STAGE%"
echo [3/4] jpackage %MAINJAR% ...
"%JPACKAGE%" --type app-image --name realt --input "%LIB%" ^
    --main-jar "%MAINJAR%" --main-class realt.app.MainKt ^
    --java-options "-XX:MaxRAMPercentage=75" --java-options "-Dfile.encoding=UTF-8" ^
    --add-modules java.base --win-console --dest "%STAGE%"
if errorlevel 1 (
    echo [ERROR] jpackage failed, see messages above.
    goto fail
)

rem ---- 4. Copy into the output folder (keeps your own files there) -------------
echo [4/4] Copy to %OUT%
if not exist "%OUT%" mkdir "%OUT%"
robocopy "%STAGE%\realt\app" "%OUT%\app" /MIR /NFL /NDL /NJH /NJS /NP >nul
if errorlevel 8 goto copyfail
robocopy "%STAGE%\realt\runtime" "%OUT%\runtime" /MIR /NFL /NDL /NJH /NJS /NP >nul
if errorlevel 8 goto copyfail
copy /y "%STAGE%\realt\*.*" "%OUT%\" >nul
if errorlevel 1 goto copyfail
copy /y "%ROOT%\README.md" "%OUT%\" >nul

echo.
"%OUT%\realt.exe" version
if errorlevel 1 (
    echo [ERROR] The packaged realt.exe did not start.
    goto fail
)
echo.
echo ==== Build OK ====
echo Run it with:
echo     cd /d "%OUT%"
echo     realt.exe version
echo     realt.exe train --sample 1000 --out model1
echo The "version" line shows the build time; it must match this build.
if exist "%ROOT%\build\pkg\realt\realt.exe" (
    echo.
    echo NOTE: %ROOT%\build\pkg\realt is an OLD build. Do not run it any more.
    echo       Move any results you want to keep from there into %OUT%.
)
endlocal
exit /b 0

:copyfail
echo [ERROR] Copy to %OUT% failed. Close any running realt.exe and try again.
:fail
endlocal
exit /b 1
