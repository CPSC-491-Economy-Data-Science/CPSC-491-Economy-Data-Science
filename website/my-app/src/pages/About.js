import { Card, CardContent, Grid, Paper, Typography } from "@material-ui/core";
import Logo from '../images/github-logo.png';

export default function About() {
    return (
    <div>
        <h1>About Us</h1>

        <p className="about-p">As a group of CS major students that all had the same thoughts for life after 
            school, we wanted to work on a project that would have us working with new 
            frameworks and technology. In this data science project, we take economic data, 
            such as housing, inflation, wages, and other economic factors and create a model 
            that can predict new data. The result of the data will provide a narrative to what 
            the data is trying to portray. This can be something about the data, such as why 
            it might be the way it is or what should be done given this new found data. The 
            data will do this specific action, because we will select datasets that are linked 
            to the economy. In order for the data to tell a narrative, we will need to go 
            through the data analyzing and data processing process first. This process includes 
            gathering, sorting and cleaning of the data, so that the data are usable in terms 
            of our specific needs. We shall create different types of models providing 
            perspective into seeing how well they perform with the given data. Our target 
            users are any economic analysis, data analysis, investors, researcher and so forth.
        </p>

        <br></br><br></br>
        <h2 className="about-h2">Created By:</h2>
        <ul className="about-names">
            <li>
                Britney Fernandez
            </li>
            <li>
                Nolan O’Donnell 
            </li>
            <li>
                Gregory Miller
            </li>
            <li>
                Kevin Tu
            </li>
        </ul>

        <br></br><br></br>
        <h2 className="visit-repo-h2">Visit our repoistory!</h2>
        <a href = "https://github.com/CPSC-491-Economy-Data-Science/CPSC-491-Economy-Data-Science">
        <img className="github-logo" src={Logo} alt="alt-logo"/></a>
    </div>
    )   
}