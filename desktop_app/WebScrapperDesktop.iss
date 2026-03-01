#define MyAppName "WebScrapper Desktop"
#ifndef MyAppVersion
  #define MyAppVersion "1.0.0"
#endif
#define MyAppPublisher "WebScrapper"
#define MyAppExeName "WebScrapperDesktop.exe"

[Setup]
AppId={{8E2D6D50-2ED5-4B89-8F26-6EF9FA7A7A65}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\WebScrapper Desktop
DefaultGroupName=WebScrapper Desktop
AllowNoIcons=yes
; InfoBeforeFile=installer_before.txt
PrivilegesRequired=admin
ArchitecturesInstallIn64BitMode=x64compatible
Compression=lzma
SolidCompression=yes
WizardStyle=modern
OutputDir=..\dist
OutputBaseFilename=WebScrapperDesktop-Setup
DisableProgramGroupPage=yes
UninstallDisplayIcon={app}\{#MyAppExeName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"; Flags: unchecked

[Files]
Source: "..\dist\WebScrapperDesktop.exe"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\WebScrapper Desktop"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\WebScrapper Desktop"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch WebScrapper Desktop"; Flags: nowait postinstall skipifsilent
