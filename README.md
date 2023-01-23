<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/KatynaSada/SparseGO_code">
    <img src="images/logoSparseGO.png" width="400" alt="Logo" >
  </a>

  <h3 align="center">SparseGO</h3>

  <p align="center">
    A VNN to predict cancer drug response using a sparse explainable neural network
    +
    <br />
    a method to predict the mechanism of action of drugs.
    <br />
    <a href="https://github.com/KatynaSada/SparseGO_code/issues">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

 <p align="center"><img src="images/network.png" width="700" alt="Logo"></p>


Artificial intelligence (AI), and specifically Deep Neural Networks (DNNs), have been successfully applied to predict drug efficacy in cancer cell lines. However, understanding how the recommendation is carried out, is still a challenge when using DNNs. An algorithm called <a href="https://pubmed.ncbi.nlm.nih.gov/33096023/">DrugCell<a> showed that by simulating the Gene Ontology structure with a DNN, each neuron in the network can be assigned an interpretation. Here we present SparseGO, a sparse explainable neural network that extends this approach in two different ways: first, by optimizing the algorithm using sparse DNNs to accommodate any type of omics data as input, and second, by fusing an eXplainable Artificial Intelligence (XAI) technique with another machine learning algorithm to systematically predict the mechanism of action (MoA) of drugs.  

This project includes instructions for:
* making predictions using a trained SparseGO network,
* training a SparseGO network,
* and using the DeepMoA method to predict the MoA of drugs.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


### Built With

*   <a href="https://www.python.org/">
      <img src="images/python.png" width="110" alt="python" >
    </a>
*   <a href="https://pytorch.org/">
      <img src="images/pytorch.png" width="105" alt="pytorch" >
    </a>
*   <a href="http://geneontology.org/">
      <img src="images/geneontology.png" width="105" alt="geneontology" >
    </a>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* npm
  ```sh
  npm install npm@latest -g
  ```

### Installation

_Below is an example of how you can instruct your audience on installing and setting up your app. This template doesn't rely on any external dependencies or services._

1. Get a free API Key at [https://example.com](https://example.com)
2. Clone the repo
   ```sh
   git clone https://github.com/your_username_/Project-Name.git
   ```
3. Install NPM packages
   ```sh
   npm install
   ```
4. Enter your API in `config.js`
   ```js
   const API_KEY = 'ENTER YOUR API';
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap

- [x] Add Changelog
- [x] Add back to top links
- [ ] Add Additional Templates w/ Examples
- [ ] Add "components" document to easily copy & paste sections of the readme
- [ ] Multi-language Support
    - [ ] Chinese
    - [ ] Spanish

See the [open issues](https://github.com/othneildrew/Best-README-Template/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an <a href="https://github.com/KatynaSada/SparseGO_code/issues">issue</a> with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

Katyna Sada - ksada@unav.es - ksada@tecnun.es

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- ACKNOWLEDGMENTS -->
## Acknowledgments
* [DrugCell](https://github.com/idekerlab/DrugCell)
* [Weights & Biases](https://www.wandb.ai/)
* [Sparse Linear](https://github.com/hyeon95y/SparseLinear)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
