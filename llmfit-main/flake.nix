{
  description = "Hundreds of models & providers. One command to find what runs on your hardware.";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        version = (builtins.fromTOML (builtins.readFile ./Cargo.toml)).workspace.package.version;
      in
      {
        packages.default = pkgs.rustPlatform.buildRustPackage {
          pname = "llmfit";
          inherit version;

          src = ./.;

          cargoLock.lockFile = ./Cargo.lock;

          # Build only the TUI binary (default workspace members exclude desktop)
          cargoBuildFlags = [ "--package" "llmfit" ];
          cargoTestFlags = [ "--package" "llmfit" "--package" "llmfit-core" ];

          meta = with pkgs.lib; {
            description = "Matches LLM models to your hardware capabilities";
            homepage = "https://github.com/AlexsJones/llmfit";
            license = licenses.mit;
            maintainers = [ ];
            mainProgram = "llmfit";
          };
        };

        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            rustc
            cargo
            rust-analyzer
            clippy
            rustfmt
          ];
        };
      });
}
