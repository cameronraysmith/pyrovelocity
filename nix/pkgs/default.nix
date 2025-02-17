{
  system,
  pkgs,
}: let
  tex = pkgs.texlive.combine {
    inherit
      (pkgs.texlive)
      scheme-small
      algorithm2e
      algorithmicx
      algorithms
      algpseudocodex
      apacite
      appendix
      caption
      cm-super
      dvipng
      framed
      git-latexdiff
      latexdiff
      latexmk
      latexpand
      multirow
      ncctools
      placeins
      rsfs
      sttools
      threeparttable
      type1cm
      vruler
      wrapfig
      xurl
      ;
  };

  extraSysPackages = with pkgs; [
    atuin
    bat
    btop
    curl
    gawk
    gh
    git
    gnugrep
    gnumake
    gnused
    gnupg
    helix
    htop
    jqp
    kubectl
    kubectx
    k9s
    lazygit
    man-db
    man-pages
    neovim
    openvscode-server
    ripgrep
    skaffold
    starship
    wget
    zellij
    zsh
  ];

  sysPackages = with pkgs;
    [
      bashInteractive
      coreutils
      cacert
      direnv
      file
      findutils
      gnutar
      gzip
      less
      libgcc
      lmodern
      nix
      procps
      tex
      time
      which
    ]
    ++ pkgs.lib.optional (pkgs.lib.elem system pkgs.shadow.meta.platforms) shadow;

  coreDevPackages = with pkgs; [
    bazelisk
    bazel-buildtools
    cue
    poetry
    poethepoet
    timoni
    tree
    yq-go
  ];

  extraDevPackages = with pkgs; [
    lmodern
    pandoc
    quarto
    tex
  ];
in {
  sysPackages = sysPackages;
  extraSysPackages = extraSysPackages;
  coreDevPackages = coreDevPackages;
  devPackages = coreDevPackages ++ extraDevPackages ++ pkgs.lib.optional (system == "x86_64-linux") pkgs.nvitop;
}
